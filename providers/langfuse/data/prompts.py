import re
import json
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableSequence
from langsmith.utils import LangSmithConflictError

from utils.langfuse import lf_get_project_prompts
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
        return "openai"  # default fallback

    model_lower = model_name.lower()

    # Anthropic models
    if any(x in model_lower for x in ["claude", "anthropic"]):
        return "anthropic"

    # OpenAI models
    if any(
        x in model_lower
        for x in [
            "gpt",
            "openai",
            "o1",
            "text-davinci",
            "text-curie",
            "text-babbage",
            "text-ada",
        ]
    ):
        return "openai"

    # Default to OpenAI for unknown models
    return "openai"


def get_model_instance(model_name: str, model_params: dict = None):
    """Get the appropriate LangChain model instance based on the model name."""
    provider = detect_model_provider(model_name)
    # Sanitize params to avoid conflicts/unexpected kwargs
    params = dict(model_params or {})
    # Remove values that would duplicate or are unsupported by LangChain clients
    params.pop("model", None)
    params.pop("supported_languages", None)

    if provider == "anthropic":
        # Anthropic models
        return ChatAnthropic(model=model_name, **params)
    else:
        # OpenAI models (default)
        # Note: ChatOpenAI uses 'model' parameter, not 'model_name'
        return ChatOpenAI(model=model_name, **params)


def langfuse_prompt_conversion(lf_prompt: dict) -> dict:
    """
    Map Langfuse prompt JSON → dict ready for push_langsmith_prompt().
    """
    # Prefer direct fields from v2 detail; fallback to nested versions if present
    pv = lf_prompt.get("latest_version", {}) or lf_prompt.get("current_version", {})

    # Description
    description = (
        lf_prompt.get("description")
        or pv.get("description")
        or ""
    )

    # Model and params (if provided). In v2, config may live top-level.
    config = (lf_prompt.get("config") or pv.get("config") or {}) or {}
    model = config.get("model") or pv.get("model") or ""
    model_params = config if isinstance(config, dict) else {}

    # Build template string
    prompt_template = ""
    # v2: top-level prompt can be a list of messages
    if isinstance(lf_prompt.get("prompt"), list):
        msgs = [f"{m.get('role','user')}: {m.get('content','')}" for m in lf_prompt.get("prompt", [])]
        prompt_template = "\n\n".join(msgs)
    elif isinstance(lf_prompt.get("prompt"), str):
        prompt_template = lf_prompt["prompt"]
    elif isinstance(pv.get("messages"), list):
        msgs = [f"{m.get('role','user')}: {m.get('content','')}" for m in pv.get("messages", [])]
        prompt_template = "\n\n".join(msgs)
    elif isinstance(pv.get("prompt"), str):
        prompt_template = pv.get("prompt")
    else:
        # Last resort: dump object for visibility
        prompt_template = json.dumps(lf_prompt)

    # Normalize mustache variants to {var}
    tpl = prompt_template
    tpl = re.sub(r"\{\{\{(\w+)\}\}\}", r"{\1}", tpl)
    tpl = re.sub(r"\{\{(\w+)\}\}", r"{\1}", tpl)

    # Detect variables after normalization
    var_pat = re.findall(r"\{(\w+)\}", tpl)

    # Name → lowercase, spaces to hyphens, robust to missing keys
    raw_name = lf_prompt.get("name") or lf_prompt.get("slug") or f"prompt-{lf_prompt.get('id','')}"
    name = str(raw_name).lower().replace(" ", "-")

    out: dict = {
        "name": name,
        "description": description,
        "prompt_template": tpl,
        "input_variables": list(sorted(set(var_pat))),
        "metadata": {
            "langfuse_id": lf_prompt.get("id"),
            "langfuse_labels": lf_prompt.get("labels", []),
            "langfuse_tags": lf_prompt.get("tags", []),
            "created": lf_prompt.get("createdAt") or lf_prompt.get("created_at") or lf_prompt.get("created"),
            "type": lf_prompt.get("type") or pv.get("type"),
            "version": lf_prompt.get("version") or pv.get("version"),
            "model": model,
            "model_params": model_params,
            "original_source": "langfuse",
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
            print(
                f"       ! failed to create model {model_name}, using prompt only: {e}"
            )
            obj = chat_prompt
    else:
        if model_name and not include_model:
            print(f"       • prompt only (model {model_name} excluded by flag)")
        obj = chat_prompt
    return obj


def migrate_prompts(workspace_id: str, project_id: str):
    prompts = lf_get_project_prompts(project_id)
    if prompts:
        print(f"    - migrating {len(prompts)} prompt(s)…")
        for lf_p in prompts:
            try:
                ls_p_dict = langfuse_prompt_conversion(lf_p)
                ls_p_obj = prompt_dict_to_obj(ls_p_dict, include_model=INCLUDE_MODEL_IN_PROMPTS)
                url = ls_push_prompt(ls_p_dict["name"], ls_p_dict["description"], ls_p_obj, workspace_id)
                pname_disp = lf_p.get("name") or lf_p.get("slug") or lf_p.get("id")
                print(f"       • {pname_disp}  →  {url}")
            except Exception as e:
                pname_disp = lf_p.get("name") or lf_p.get("slug") or lf_p.get("id")
                if isinstance(e, LangSmithConflictError):
                    print(f"       • prompt '{pname_disp}' already exists, skipping...")
                    continue
                print(f"       x prompt '{pname_disp}' failed: {e}")
                traceback.print_exc()
    else:
        print("    (no prompts)")