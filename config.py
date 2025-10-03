import os
from dotenv import load_dotenv

load_dotenv()

LF_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LF_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LF_BASE = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

LS_API_KEY = os.getenv("LANGSMITH_API_KEY")
LS_ORG_ID = os.environ["LANGSMITH_ORGANIZATION_ID"]
LS_BASE = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

LF_HEADERS = {
    "Content-Type": "application/json",
}

LS_HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": LS_API_KEY,
    "X-Organization-Id": LS_ORG_ID,
}

# Migration settings
INCLUDE_MODEL_IN_PROMPTS = (
    os.getenv("INCLUDE_MODEL_IN_PROMPTS", "true").lower() == "true"
)
NUM_TRACES_TO_REPLAY = (
    int(os.getenv("NUM_TRACES_TO_REPLAY", "0"))
)
