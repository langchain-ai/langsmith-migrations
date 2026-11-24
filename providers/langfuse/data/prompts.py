import traceback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence

from utils.langfuse import lf_get_prompt_history
from utils.langsmith import ls_prompt_exists, ls_push_prompt_version
from config import INCLUDE_MODEL_IN_PROMPTS

# Langfuse adds this label itself to the newest version. It is not a deployment
# label, so it is not carried over as a LangSmith commit tag.
_AUTO_LABELS = {"latest"}

# Langfuse chat roles -> LangChain message roles.
_ROLE_MAP = {
    "system": "system",
    "developer": "system",
    "user": "human",
    "human": "human",
    "assistant": "ai",
    "ai": "ai",
}


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

    # Imported here so that prompts can be migrated without the provider
    # packages when INCLUDE_MODEL_IN_PROMPTS=false.
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, **params)

    from langchain_openai import ChatOpenAI

    # ChatOpenAI uses 'model' parameter, not 'model_name'
    return ChatOpenAI(model=model_name, **params)


def langsmith_prompt_name(lf_name: str) -> str:
    """Name a Langfuse prompt gets in LangSmith.

    Lowercased, spaces become hyphens. Langfuse folder names ("team/prompt") use
    "/", which LangSmith reads as owner/name, so it becomes "-" too.
    """
    return str(lf_name).lower().replace(" ", "-").replace("/", "-")


def build_chat_prompt(lf_prompt: dict) -> ChatPromptTemplate:
    """Build a LangSmith-ready ChatPromptTemplate from one Langfuse prompt version.

    - Every line of every message is kept (the previous string round trip dropped
      any line that did not start with "role:").
    - Langfuse's {{variable}} syntax is kept as is. LangSmith supports mustache
      templates, so literal braces such as JSON examples stay literal.
    - Chat placeholders become optional MessagesPlaceholders.
    """
    raw = lf_prompt.get("prompt")
    messages: list = []

    if isinstance(raw, str):
        messages.append(("human", raw))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "placeholder":
                messages.append(MessagesPlaceholder(item["name"], optional=True))
                continue
            role = _ROLE_MAP.get(str(item.get("role", "user")).lower())
            if role is None:
                print(
                    f"       ! unknown role '{item.get('role')}' in prompt "
                    f"'{lf_prompt.get('name')}', treating it as 'human'"
                )
                role = "human"
            messages.append((role, item.get("content", "")))
    else:
        raise ValueError(
            f"prompt '{lf_prompt.get('name')}' version {lf_prompt.get('version')} "
            "has no text or chat content"
        )

    return ChatPromptTemplate.from_messages(messages, template_format="mustache")


def prompt_to_obj(lf_prompt: dict, include_model: bool = True) -> object:
    """Prompt object to push: the template alone, or template | model."""
    chat_prompt = build_chat_prompt(lf_prompt)
    config = lf_prompt.get("config") or {}
    model_name = config.get("model") if isinstance(config, dict) else None

    if model_name and include_model:
        try:
            model = get_model_instance(model_name, config)
            print(f"       ... using {detect_model_provider(model_name)} model: {model_name}")
            return RunnableSequence(chat_prompt, model)
        except Exception as e:
            print(f"       ! failed to create model {model_name}, using prompt only: {e}")
    elif model_name:
        print(f"       • prompt only (model {model_name} excluded by flag)")
    return chat_prompt


def migrate_prompts(workspace_id: str, project_id: str):
    """Migrate every version of every prompt, oldest first, so LangSmith ends up
    with the same commit history. Langfuse labels become commit tags.

    A prompt that already exists in LangSmith is skipped, so the script can be
    re-run without adding duplicate commits.
    """
    history = lf_get_prompt_history(project_id)
    if not history:
        print("    (no prompts)")
        return

    print(f"    - migrating {len(history)} prompt(s)…")
    for entry in history:
        lf_name = entry["name"]
        ls_name = langsmith_prompt_name(lf_name)
        try:
            if entry.get("failed_versions"):
                print(
                    f"       x prompt '{lf_name}' skipped: could not fetch "
                    f"version(s) {entry['failed_versions']}; migrating the rest would leave gaps in its history"
                )
                continue
            if ls_prompt_exists(ls_name, workspace_id):
                print(f"       • prompt '{lf_name}' already exists as '{ls_name}', skipping...")
                continue

            versions = sorted(entry["versions"], key=lambda v: v.get("version") or 0)
            for lf_p in versions:
                number = lf_p.get("version")
                labels = [l for l in (lf_p.get("labels") or []) if l not in _AUTO_LABELS]
                obj = prompt_to_obj(lf_p, include_model=INCLUDE_MODEL_IN_PROMPTS)
                message = lf_p.get("commitMessage") or ""
                url, applied, failed_tags = ls_push_prompt_version(
                    ls_name,
                    lf_p.get("description") or "",
                    obj,
                    workspace_id,
                    tags=lf_p.get("tags") or None,
                    commit_tags=labels or None,
                    commit_description=(
                        f"Migrated from Langfuse version {number}"
                        + (f": {message}" if message else "")
                    )[:1000],
                )
                print(f"       • {lf_name} v{number} (tags: {applied or 'none'})  →  {url}")
                for tag, reason in failed_tags:
                    print(f"         ! could not apply tag '{tag}' to this commit: {reason}")
            print(f"         {lf_name}  =>  LangSmith prompt '{ls_name}'")
        except Exception as e:
            print(
                f"       x prompt '{lf_name}' failed: {e}\n"
                f"         If '{ls_name}' was partly created in LangSmith, delete it before re-running."
            )
            traceback.print_exc()
