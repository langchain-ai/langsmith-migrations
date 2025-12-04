import os
import requests
from dotenv import load_dotenv
import phoenix as px
from phoenix.client import Client

load_dotenv()

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
PHOENIX_SPACE = os.getenv("PHOENIX_SPACE")
PHOENIX_BASE = "https://app.phoenix.arize.com"

# Build the base URL with space if provided
PHOENIX_BASE_URL = f"{PHOENIX_BASE}/s/{PHOENIX_SPACE}" if PHOENIX_SPACE else PHOENIX_BASE

# Set environment variables for px.Client() authentication
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = PHOENIX_BASE_URL


def get_phoenix_client() -> Client:
    """Get an authenticated Phoenix client (for datasets, prompts, projects)."""
    return Client(
        base_url=PHOENIX_BASE_URL,
        api_key=PHOENIX_API_KEY,
    )


def get_px_client():
    """Get a px.Client() for spans/traces (has get_spans_dataframe)."""
    return px.Client()


def phoenix_get_projects() -> list[dict]:
    """List Phoenix projects."""
    client = get_phoenix_client()
    projects = client.projects.list()
    return [{"name": p["name"]} for p in projects]


def phoenix_get_traces(project_name: str, limit: int = 100) -> dict:
    """Fetch traces from a Phoenix project, grouped by trace_id.
    
    Returns: dict[trace_id -> list of span dicts]
    """
    client = get_phoenix_client()
    
    # Get all spans as DataFrame
    spans_df = client.spans.get_spans_dataframe(project_name=project_name)
    
    if spans_df is None or spans_df.empty:
        return {}
    
    # Group spans into traces
    traces = {}
    for trace_id, group in spans_df.groupby("context.trace_id"):
        sorted_group = group.sort_values("start_time")
        traces[trace_id] = sorted_group.to_dict(orient='records')
        
        # Limit to N traces
        if len(traces) >= limit:
            break
    
    return traces


def phoenix_get_datasets() -> list[dict]:
    """List datasets in Phoenix."""
    client = get_phoenix_client()
    datasets = client.datasets.list()
    return [{"id": d["id"], "name": d["name"], "description": d.get("description", "")} for d in datasets]


def phoenix_get_dataset_examples(dataset_id: str = None, dataset_name: str = None) -> list[dict]:
    """Fetch examples from a Phoenix dataset."""
    client = get_phoenix_client()
    
    if dataset_name: 
        dataset = client.datasets.get_dataset(dataset=dataset_name)
    else:
        return []
    
    # Get examples from _examples_data.examples attribute
    if hasattr(dataset, '_examples_data') and dataset._examples_data is not None:
        examples_data = dataset._examples_data
        if hasattr(examples_data, 'examples') and examples_data.examples is not None:
            examples = examples_data.examples
        elif isinstance(examples_data, dict) and 'examples' in examples_data:
            examples = examples_data['examples']
        else:
            return []
        
        if hasattr(examples, 'to_dict'):
            return examples.to_dict(orient='records')
        if isinstance(examples, list):
            return examples
        return list(examples)
    
    return []


def phoenix_get_prompts() -> list[dict]:
    """List prompts in Phoenix using REST API. Phoenix Client does not support listing prompts."""
    headers = {
        "Authorization": f"Bearer {PHOENIX_API_KEY}",
        "api_key": PHOENIX_API_KEY,
        "x-api-key": PHOENIX_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.get(f"{PHOENIX_BASE_URL}/v1/prompts", headers=headers)
        
        if resp.status_code != 200:
            print(f"    x Failed to fetch prompts: {resp.status_code} {resp.text}")
            return []
        
        data = resp.json()
        prompts = data.get('data', []) if isinstance(data, dict) else data
        
        result = []
        for p in prompts:
            if isinstance(p, dict):
                result.append({
                    "id": p.get("id", ""),
                    "name": p.get("name", ""),
                    "description": p.get("description", "") or ""
                })
        return result
    except Exception as e:
        print(f"    x Error fetching prompts via REST: {e}")
        return []


def phoenix_get_prompt(prompt_name: str):
    """Get a specific prompt by name."""
    client = get_phoenix_client()
    return client.prompts.get(prompt_identifier=prompt_name)
