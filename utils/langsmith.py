import requests
from typing import Optional

from langsmith import Client
from config import LS_BASE, LS_HEADERS, LS_API_KEY


def ls_get_or_create_workspace(ws_name: str) -> dict:
    try:
        r = requests.post(
            f"{LS_BASE}/api/v1/workspaces",
            headers=LS_HEADERS,
            json={"display_name": ws_name},
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        # fetch existing
        ex = requests.get(f"{LS_BASE}/api/v1/workspaces", headers=LS_HEADERS).json()
        return next(ws for ws in ex if ws["display_name"] == ws_name)


def ls_create_dataset(workspace_id: str, name: str) -> str:
    hdrs = LS_HEADERS | {"X-Tenant-Id": workspace_id}
    r = requests.post(f"{LS_BASE}/api/v1/datasets", headers=hdrs, json={"name": name})
    if r.status_code == 409:  # already there
        r = requests.get(f"{LS_BASE}/api/v1/datasets", headers=hdrs).json()
        return next(ds["id"] for ds in r if ds["name"] == name)
    r.raise_for_status()
    return r.json()["id"]


def ls_upload_examples(workspace_id: str, dataset_id: str, examples: list[dict]):
    hdrs = LS_HEADERS | {"X-Tenant-Id": workspace_id}
    for ex in examples:
        payload = {"dataset_id": dataset_id} | ex
        requests.post(
            f"{LS_BASE}/api/v1/examples", headers=hdrs, json=payload
        ).raise_for_status()


def ls_push_prompt(
    name: str,
    description: str,
    prompt_obj: object,
    workspace_id: str,
    pat: Optional[str] = None,
) -> str:
    """Create/Update a prompt (+model) in the given workspace; return URL."""
    pat = pat or LS_API_KEY
    session = requests.Session()
    session.headers.update({"X-Tenant-Id": workspace_id})
    client = Client(api_key=pat, api_url=LS_BASE, session=session)
    
    url = client.push_prompt(
        name,
        object=prompt_obj,
        description=description,
        # metadata & input vars not yet supported in SDK call → store via tags
    )
    return url


def ls_upload_runs(workspace_id: str, runs: list[dict]):
    """Upload runs and feedback to LangSmith workspace via REST API.

    Expects run dicts shaped similarly to LangSmith runs. Feedback items can be
    passed as dicts with key 'type' == 'feedback'.
    """
    hdrs = LS_HEADERS | {"X-Tenant-Id": workspace_id}
    for item in runs:
        if isinstance(item, dict) and item.get("type") == "feedback":
            payload = {
                "run_id": item.get("run_id"),
                "key": item.get("key"),
                "score": item.get("score"),
                "comment": item.get("comment"),
                "source": item.get("source"),
                "metadata": item.get("metadata", {}),
                "timestamp": item.get("timestamp"),
            }
            requests.post(f"{LS_BASE}/api/v1/feedback", headers=hdrs, json=payload).raise_for_status()
            continue

        # Normal run
        run_payload = {k: v for k, v in item.items() if k in {
            "id","name","run_type","inputs","outputs","start_time","end_time","session_id",
            "parent_run_id","tags","metadata","error","invocation_params","usage_metadata"
        }}
        requests.post(f"{LS_BASE}/api/v1/runs", headers=hdrs, json=run_payload).raise_for_status()
