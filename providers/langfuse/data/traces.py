import os
import sys
import uuid
import datetime as dt
import time
import json
from types import SimpleNamespace
from dotenv import load_dotenv
from config import NUM_TRACES_TO_REPLAY
from utils.langsmith import ls_replay_runs_sdk
from utils.langfuse import lf_get
 
load_dotenv()

def safe_isoformat(dt_obj):
    if dt_obj is None:
        return None
    if not isinstance(dt_obj, dt.datetime):
        if isinstance(dt_obj, str):
            try:
                dt.datetime.fromisoformat(dt_obj.replace('Z', '+00:00'))
                return dt_obj
            except ValueError:
                return None
        return None
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    s = dt_obj.isoformat(timespec='milliseconds')
    return s[:-6] + 'Z' if s.endswith('+00:00') else s

def _compact_ts(ts_val):
    if ts_val is None:
        return ""
    if isinstance(ts_val, str):
        try:
            s = ts_val[:-1] + '+00:00' if ts_val.endswith('Z') else ts_val
            dt_obj = dt.datetime.fromisoformat(s)
        except Exception:
            return ""
    else:
        dt_obj = ts_val
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return dt_obj.strftime('%Y%m%dT%H%M%S') + f"{dt_obj.microsecond:06d}" + 'Z'


# ------------ Normalization helpers ------------

def _span_type_mapping(span_type: str) -> str:
    mapping = {
        "SPAN": "chain",
        "AGENT": "chain",
        "EVENT": "chain",
        "GENERATION": "llm",
        "TOOL": "tool",
        "RETRIEVER": "retriever",
        "EVALUATOR": "chain",
        "EMBEDDING": "llm",
        "GUARDRAIL": "chain"
    }
    if span_type in mapping:
        return mapping[span_type]
    return "chain"

def _to_plain(val):
    if isinstance(val, SimpleNamespace):
        return {k: _to_plain(getattr(val, k)) for k in vars(val)}
    if isinstance(val, (list, tuple, set)):
        return [_to_plain(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_plain(v) for k, v in val.items()}
    return val


def _to_messages(obj, default_role="user"):
    obj = _to_plain(obj)
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        msgs = []
        for m in obj["messages"]:
            if isinstance(m, dict) and {"role", "content"}.issubset(m.keys()):
                msgs.append({"role": m.get("role") or default_role, "content": m.get("content")})
            elif isinstance(m, str):
                msgs.append({"role": default_role, "content": m})
        return {"messages": msgs}
    if isinstance(obj, dict) and {"role", "content"}.issubset(obj.keys()):
        return {"messages": [{"role": obj.get("role") or default_role, "content": obj.get("content")}]}
    if isinstance(obj, str):
        return {"messages": [{"role": default_role, "content": obj}]}
    if isinstance(obj, list):
        msgs = []
        for m in obj:
            if isinstance(m, dict) and {"role", "content"}.issubset(m.keys()):
                msgs.append({"role": m.get("role") or default_role, "content": m.get("content")})
            elif isinstance(m, str):
                msgs.append({"role": default_role, "content": m})
        if msgs:
            return {"messages": msgs}
    return {"messages": []}


def _ensure_end_times(runs: list[dict]):
    for r in runs:
        if isinstance(r, dict) and r.get("end_time") is None:
            r["end_time"] = r.get("start_time")


def _children_map(runs: list[dict]) -> dict:
    id_to_run = {r["id"]: r for r in runs if isinstance(r, dict)}
    cmap: dict[str, list[str]] = {}
    for r in runs:
        if not isinstance(r, dict):
            continue
        pid = r.get("parent_run_id")
        if pid:
            cmap.setdefault(pid, []).append(r["id"])
    for pid, kids in list(cmap.items()):
        kids.sort(key=lambda k: id_to_run.get(k, {}).get('start_time') or '')
        cmap[pid] = kids
    return cmap


def _assign_dotted_order(runs: list[dict], root_id: str):
    id_to_run = {r["id"]: r for r in runs if isinstance(r, dict)}
    cmap = _children_map(runs)
    def assign(run_id: str, parent_dotted: str | None):
        run = id_to_run.get(run_id)
        if not run:
            return
        ts = run.get('start_time') or run.get('end_time') or id_to_run.get(root_id, {}).get('start_time')
        seg = _compact_ts(ts) + run_id
        dotted = seg if not parent_dotted else f"{parent_dotted}.{seg}"
        run["dotted_order"] = dotted
        for kid in cmap.get(run_id, []):
            assign(kid, dotted)
    assign(root_id, None)
    # root end_time to max child
    max_child = None
    for cid in cmap.get(root_id, []):
        child = id_to_run.get(cid)
        if child and child.get("end_time"):
            s = child["end_time"]
            try:
                s = s[:-1] + '+00:00' if isinstance(s, str) and s.endswith('Z') else s
                dt_obj = dt.datetime.fromisoformat(s) if isinstance(s, str) else s
                if not max_child or dt_obj > max_child:
                    max_child = dt_obj
            except Exception:
                continue
    root = id_to_run.get(root_id)
    if root and max_child:
        if max_child.tzinfo is None:
            max_child = max_child.replace(tzinfo=dt.timezone.utc)
        root["end_time"] = max_child.strftime('%Y-%m-%dT%H:%M:%S') + f".{max_child.microsecond:06d}Z"


def _get_attr(obj, names: list[str]):
    for n in names:
        v = None
        if isinstance(obj, dict):
            v = obj.get(n)
        else:
            try:
                v = getattr(obj, n)
            except Exception:
                v = None
        if v is not None:
            return v
    return None


def map_langfuse_to_langsmith(source_trace):
    runs: list[dict] = []
    # Root
    new_root_id = str(uuid.uuid4())
    root = {
        "id": new_root_id,
        "trace_id": new_root_id,
        "name": _get_attr(source_trace, ['name']) or "Trace",
        "run_type": "chain",
        "session_id": _get_attr(source_trace, ['session_id','sessionId']),
        "tags": _get_attr(source_trace, ['tags']) or [],
        "metadata": _get_attr(source_trace, ['metadata']) if isinstance(_get_attr(source_trace, ['metadata']), dict) else {},
        "inputs": _get_attr(source_trace, ['input','inputs']),
        "outputs": _get_attr(source_trace, ['output','outputs']),
        "start_time": safe_isoformat(_get_attr(source_trace, ['timestamp','start_time','startTime'])),
        "end_time": safe_isoformat(_get_attr(source_trace, ['end_time','endTime'])),
        "status": "completed",
        "parent_run_id": None,
    }
    runs.append(root)
    # Children: two-pass to preserve nesting (parent before child lookup)
    obs_list = _get_attr(source_trace, ['observations']) or []
    # First pass: assign stable new run IDs for each observation
    obs_id_to_run_id: dict[str, str] = {}
    for obs in obs_list:
        lf_obs_id = _get_attr(obs, ['id']) or str(uuid.uuid4())
        obs_id_to_run_id[str(lf_obs_id)] = str(uuid.uuid4())
    # Second pass: build runs in temporal order and set parent_run_id from mapping
    for obs in sorted(obs_list, key=lambda o: _get_attr(o, ['start_time','startTime']) or ''):
        run_id = obs_id_to_run_id[str(_get_attr(obs, ['id']) or '')]
        obs_type = (_get_attr(obs, ['type']) or '').upper()
        run_type = _span_type_mapping(obs_type)
        inputs = _to_plain(_get_attr(obs, ['input','inputs']))
        outputs = _to_plain(_get_attr(obs, ['output','outputs']))
        if run_type == "llm" and _get_attr(obs, ['model']):
            inputs = _to_messages(inputs, default_role="user")
            outputs = _to_messages(outputs, default_role="assistant")
        parent_obs_id = _get_attr(obs, ['parent_observation_id','parentObservationId'])
        parent_run_id = obs_id_to_run_id.get(str(parent_obs_id)) if parent_obs_id else root["id"]
        run = {
            "id": run_id,
            "trace_id": root["id"],
            "name": _get_attr(obs, ['name']) or f"{str(_get_attr(obs, ['type']) or 'obs').lower()}_{_get_attr(obs, ['id']) or ''}",
            "run_type": run_type,
            "parent_run_id": parent_run_id,
            "session_id": _get_attr(source_trace, ['session_id','sessionId']),
            "tags": [],
            "metadata": _get_attr(obs, ['metadata']) if isinstance(_get_attr(obs, ['metadata']), dict) else {},
            "inputs": inputs,
            "outputs": outputs,
            "start_time": safe_isoformat(_get_attr(obs, ['start_time','startTime'])),
            "end_time": safe_isoformat(_get_attr(obs, ['end_time','endTime'])) if _get_attr(obs, ['end_time','endTime']) else None,
            "status": "completed",
        }
        if run_type == "llm" and _get_attr(obs, ['model']):
            model_lower = str(_get_attr(obs, ['model'])).lower()
            run.setdefault("metadata", {})
            run["metadata"]["ls_provider"] = "openai" if "openai" in model_lower else ("anthropic" if "anthropic" in model_lower or "claude" in model_lower else ("google" if "google" in model_lower else "unknown"))
        runs.append(run)
    # Finish: end times + dotted order
    _ensure_end_times(runs)
    _assign_dotted_order(runs, root["id"])
    return runs


def fetch_and_transform_traces(workspace_id: str, sleep_between_gets=0.7, max_retries=4):
    """
    Fetch most recent traces using Public API and transform them into ingestion events.
    Enforces NUM_TRACES_TO_REPLAY as a hard cap.
    """
    try:
        max_traces = int(NUM_TRACES_TO_REPLAY)
    except Exception:
        max_traces = None

    page = 1
    limit = 50
    total_processed = 0
    total_failed_fetch = 0
    total_failed_transform = 0

    accumulated_runs: list[dict] = []
    while True:
        try:
            listing = lf_get("/api/public/traces", page=page, limit=limit)
        except Exception as e:
            print(f"Error fetching trace list page {page}: {e}")
            break

        objs = (
            listing.get("objects") if isinstance(listing, dict) and "objects" in listing
            else listing.get("data") if isinstance(listing, dict) else listing
        ) or []
        if not objs:
            break

        for item in objs:
            if max_traces is not None and total_processed >= max_traces:
                break
            source_trace_id = item.get("id")
            if not source_trace_id:
                continue

            # Fetch full trace details with retry (Public API)
            source_detail = None
            fetch_detail_success = False
            detail_retries = 0
            while not fetch_detail_success and detail_retries < max_retries:
                time.sleep(sleep_between_gets * (2 ** detail_retries))
                try:
                    source_detail = lf_get(f"/api/public/traces/{source_trace_id}")
                    fetch_detail_success = True
                except Exception:
                    detail_retries += 1
                    if detail_retries >= max_retries:
                        total_failed_fetch += 1
                        break

            if not fetch_detail_success or not isinstance(source_detail, dict):
                continue

            try:
                runs_batch = map_langfuse_to_langsmith(source_detail)
                if not runs_batch:
                    total_failed_transform += 1
                    continue
                # Append to runs accumulator
                accumulated_runs.extend(runs_batch)
                total_processed += 1
                if max_traces is not None and total_processed >= max_traces:
                    break
            except Exception:
                total_failed_transform += 1
                continue

        if max_traces is not None and total_processed >= max_traces:
            break
        page += 1

    # Upload accumulated runs to LangSmith workspace (SDK only)
    if accumulated_runs:
        ls_replay_runs_sdk(workspace_id, accumulated_runs)

    print(f"        • Processed traces: {total_processed}")
    print(f"        • Failed fetching details (after retries): {total_failed_fetch}")
    print(f"        • Failed transforming data (incl. skipping): {total_failed_transform}")

def migrate_traces(workspace_id: str, project_id: str):
    print(f"    - migrating recent traces…")
    fetch_and_transform_traces(workspace_id)