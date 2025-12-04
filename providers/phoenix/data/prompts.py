import re
import json
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableSequence
from langsmith.utils import LangSmithConflictError

from utils.phoenix import phoenix_get_prompts, phoenix_get_prompt
from utils.langsmith import ls_push_prompt
from config import INCLUDE_MODEL_IN_PROMPTS


def string_to_chat_template(template: str) -> ChatPromptTemplate:
    """Detect 'role: content' lines and build a ChatPromptTemplate; fall back."""
    lines = [l.strip() for l in template.splitlines() if l.strip()]
    pairs: list[tuple[str, str]] = []
    for line in lines:
        m = re.match(r"^(system|user|assistant|tool):\s*(.*)$", line, re.I)
        if m:
            pairs.append((m.group(1).lower(), m.group(2)))
    if pairs:
        return ChatPromptTemplate.from_messages(pairs)
    return ChatPromptTemplate.from_template(template)


def detect_model_provider(model_name: str) -> str:
    """Detect if a model is from OpenAI, Anthropic, or other provider."""
    if not model_name:
        return "openai"

    model_lower = model_name.lower()

    if any(x in model_lower for x in ["claude", "anthropic"]):
        return "anthropic"

    if any(
        x in model_lower
        for x in ["gpt", "openai", "o1", "text-davinci", "text-curie", "text-babbage", "text-ada"]
    ):
        return "openai"

    return "openai"


def get_model_instance(model_name: str, model_params: dict = None):
    """Get the appropriate LangChain model instance based on the model name."""
    provider = detect_model_provider(model_name)
    params = dict(model_params or {})
    
    # Handle Phoenix nested invocation_parameters format:
    # {'type': 'openai', 'openai': {'temperature': 1.0}}
    if 'type' in params:
        param_type = params.pop('type', None)
        # Extract actual params from nested provider key
        if param_type and param_type in params:
            nested_params = params.pop(param_type, {})
            params.update(nested_params)
        # Also try 'openai' or 'anthropic' keys
        for key in ['openai', 'anthropic']:
            if key in params:
                nested_params = params.pop(key, {})
                params.update(nested_params)
    
    # Remove invalid keys
    params.pop("model", None)
    params.pop("supported_languages", None)

    if provider == "anthropic":
        return ChatAnthropic(model=model_name, **params)
    else:
        return ChatOpenAI(model=model_name, **params)


def phoenix_prompt_conversion(phoenix_prompt, prompt_info: dict = None) -> dict:
    """
    Map Phoenix prompt → dict ready for push_langsmith_prompt().
    
    Args:
        phoenix_prompt: PromptVersion object or dict
        prompt_info: Original prompt info dict with name/description from list
    """
    prompt_info = prompt_info or {}
    
    # Extract data from PromptVersion object's internal attributes
    d = phoenix_prompt.__dict__ if hasattr(phoenix_prompt, '__dict__') else {}
    
    # Get template/messages
    template_data = d.get('_template', {})
    messages_list = template_data.get('messages', []) if isinstance(template_data, dict) else []
    
    # Extract model info
    model = d.get('_model_name', '')
    model_provider = d.get('_model_provider', '')
    template_format = d.get('_template_format', '')  # MUSTACHE, JINJA, etc.
    invocation_params = d.get('_invocation_parameters', {}) or {}
    
    # Description from PromptVersion or prompt_info
    description = d.get('_description', '') or prompt_info.get("description", "")
    
    # Convert messages to template string
    # Phoenix messages have nested content: [{"type": "text", "text": "..."}]
    template = ""
    if messages_list:
        parts = []
        for msg in messages_list:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content_parts = msg.get('content', [])
                
                # Handle nested content array
                if isinstance(content_parts, list):
                    text_parts = []
                    for cp in content_parts:
                        if isinstance(cp, dict) and cp.get('type') == 'text':
                            text_parts.append(cp.get('text', ''))
                        elif isinstance(cp, str):
                            text_parts.append(cp)
                    content = ''.join(text_parts)
                elif isinstance(content_parts, str):
                    content = content_parts
                else:
                    content = str(content_parts)
                
                parts.append(f"{role}: {content}")
        template = "\n\n".join(parts)
    
    # Build prompt_dict
    prompt_dict = {
        "id": d.get('_id') or prompt_info.get("id"),
        "name": prompt_info.get("name"),
        "description": description,
    }
    
    # Normalize mustache/jinja variants to {var}
    tpl = str(template)
    tpl = re.sub(r"\{\{\{(\w+)\}\}\}", r"{\1}", tpl)
    tpl = re.sub(r"\{\{(\w+)\}\}", r"{\1}", tpl)
    tpl = re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{\1}", tpl)
    
    # Detect variables
    var_pat = re.findall(r"\{(\w+)\}", tpl)
    
    # Name normalization
    raw_name = prompt_dict.get("name") or f"prompt-{prompt_dict.get('id', '')}"
    name = str(raw_name).lower().replace(" ", "-")
    
    out: dict = {
        "name": name,
        "description": description,
        "prompt_template": tpl,
        "input_variables": list(sorted(set(var_pat))),
        "metadata": {
            "phoenix_id": prompt_dict.get("id"),
            "model": model,
            "model_provider": model_provider,
            "model_params": invocation_params,
            "template_format": template_format,
            "original_source": "phoenix",
        },
    }
    return out


def prompt_dict_to_obj(prompt_dict: dict, include_model: bool = True) -> object:
    chat_prompt = string_to_chat_template(prompt_dict["prompt_template"])
    model_name = prompt_dict["metadata"].get("model")
    model_params = prompt_dict["metadata"].get("model_params", {})

    if model_name and include_model:
        try:
            model = get_model_instance(model_name, model_params)
            obj = RunnableSequence(chat_prompt, model)
            provider = detect_model_provider(model_name)
            print(f"       ... using {provider} model: {model_name}")
        except Exception as e:
            print(f"       ! failed to create model {model_name}, using prompt only: {e}")
            obj = chat_prompt
    else:
        if model_name and not include_model:
            print(f"       • prompt only (model {model_name} excluded by flag)")
        obj = chat_prompt
    return obj

#NOTE: this does not support versioning or preserving history of the prompts, but this can be done in LangSmith
def migrate_prompts(workspace_id: str):
    try:
        prompts = phoenix_get_prompts()
    except Exception as e:
        print(f"    x failed to fetch Phoenix prompts: {e}")
        traceback.print_exc()
        return

    if prompts:
        print(f"    - migrating {len(prompts)} prompt(s)…")
        for prompt_info in prompts:
            # prompt_info is dict with {id, name, description} from REST API list
            prompt_name = prompt_info.get("name")
            try:
                # Get full prompt details (PromptVersion object with template)
                full_prompt = None
                if prompt_name:
                    try:
                        full_prompt = phoenix_get_prompt(prompt_name)
                    except Exception as e:
                        print(f"       ! could not get full prompt: {e}")
                
                # Pass both: full_prompt (has template) and prompt_info (has name/description)
                ls_p_dict = phoenix_prompt_conversion(full_prompt or prompt_info, prompt_info=prompt_info)
                ls_p_obj = prompt_dict_to_obj(ls_p_dict, include_model=INCLUDE_MODEL_IN_PROMPTS)
                url = ls_push_prompt(ls_p_dict["name"], ls_p_dict["description"], ls_p_obj, workspace_id)
                print(f"       • {prompt_name}  →  {url}")
            except Exception as e:
                if isinstance(e, LangSmithConflictError):
                    print(f"       • prompt '{prompt_name}' already exists, skipping...")
                    continue
                print(f"       x prompt '{prompt_name}' failed: {e}")
                traceback.print_exc()
    else:
        print("    (no prompts)")
