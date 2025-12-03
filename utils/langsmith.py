from __future__ import annotations

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


def ls_create_dataset(workspace_id: str, name: str) -> str | None:
    """Create a dataset or return existing one's ID. Returns None if already exists."""
    hdrs = LS_HEADERS | {"X-Tenant-Id": workspace_id}
    r = requests.post(f"{LS_BASE}/api/v1/datasets", headers=hdrs, json={"name": name})
    if r.status_code == 409:  # already exists
        # Try to find existing dataset by name
        datasets = requests.get(f"{LS_BASE}/api/v1/datasets", headers=hdrs).json()
        existing = next((ds["id"] for ds in datasets if ds["name"] == name), None)
        return existing  # None if not found (skip migration for this dataset)
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



def ls_replay_runs_sdk(workspace_id: str, runs: list[dict], project_name: str = None):
    """Replay runs using LangSmith Python SDK create_run/update_run with proper ordering.

    - Creates runs (root first, then children) using create_run
    - Updates runs with end_time/outputs/error using update_run
    
    Args:
        workspace_id: LangSmith workspace ID.
        runs: List of run dicts to upload.
        project_name: LangSmith project name for traces (default: "default").
    """
    session = requests.Session()
    session.headers.update({"X-Tenant-Id": workspace_id})
    client = Client(api_key=LS_API_KEY, api_url=LS_BASE, session=session)

    def _parse_dt(val):
        if not val:
            return None
        if isinstance(val, str):
            try:
                s = val
                if s.endswith('Z'):
                    s = s[:-1] + '+00:00'
                from datetime import datetime
                return datetime.fromisoformat(s)
            except Exception:
                return None
        return val

    # Partition feedback vs runs
    normal_runs = [r for r in runs if isinstance(r, dict) and r.get("type") != "feedback"]
    feedbacks = [r for r in runs if isinstance(r, dict) and r.get("type") == "feedback"]

    # Sort by presence of parent (root first), then by start_time
    def _key(r):
        return (1 if r.get("parent_run_id") else 0, r.get("start_time") or "")
    normal_runs.sort(key=_key)

    # First pass: create_run
    for r in normal_runs:
        create_params = {
            "id": r.get("id"),
            "name": r.get("name") or "Run",
            "run_type": r.get("run_type") or "chain",
            "inputs": r.get("inputs") or {},
            "start_time": _parse_dt(r.get("start_time")),
            "tags": r.get("tags") or [],
            "trace_id": r.get("trace_id") or r.get("id"),
            "parent_run_id": r.get("parent_run_id") or r.get("parent_id"),
            "dotted_order": r.get("dotted_order"),
            "extra": {"metadata": r.get("metadata") or {},
                       "invocation_params": r.get("invocation_params") or {},
                       "usage_metadata": r.get("usage_metadata") or {}},
        }
        # Set project_name to organize traces under the source project name 
        if project_name:
            create_params["project_name"] = project_name
        # Remove empty extra
        if not any(create_params["extra"].values()):
            create_params.pop("extra")
        # Prune None
        create_params = {k: v for k, v in create_params.items() if v is not None}
        client.create_run(**create_params)

    # Second pass: update_run (end_time, outputs, error)
    for r in normal_runs:
        outputs_val = r.get("outputs")
        update_params = {
            "run_id": r.get("id"),
            "end_time": _parse_dt(r.get("end_time") or r.get("start_time")),
            "outputs": outputs_val if isinstance(outputs_val, dict) else ( {"output": outputs_val} if outputs_val is not None else {} ),
            "error": r.get("error"),
            "trace_id": r.get("trace_id") or r.get("id"),
            "dotted_order": r.get("dotted_order"),
            "parent_run_id": r.get("parent_run_id") or r.get("parent_id"),
        }
        update_params = {k: v for k, v in update_params.items() if v is not None}
        client.update_run(**update_params)

    # Feedbacks
    for fb in feedbacks:
        client.create_feedback(
            run_id=fb.get("run_id"),
            key=fb.get("key"),
            score=fb.get("score"),
            comment=fb.get("comment"),
            source=fb.get("source"),
            metadata=fb.get("metadata") or {},
        )
    
    # Explicit flush to ensure all data is sent
    try:
        if hasattr(client, 'flush'):
            client.flush()
    except Exception as e:
        print(f"       ! Flush error: {e}")
        raise
