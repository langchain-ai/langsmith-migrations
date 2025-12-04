from __future__ import annotations

import traceback
import uuid
import json
import datetime as dt
from config import NUM_TRACES_TO_REPLAY
from utils.phoenix import phoenix_get_traces
from utils.langsmith import ls_replay_runs_sdk


def safe_isoformat(dt_obj):
    if dt_obj is None:
        return None
    if isinstance(dt_obj, str):
        return dt_obj
    if not isinstance(dt_obj, dt.datetime):
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


def _span_kind_to_run_type(span_kind: str) -> str:
    if not span_kind:
        return "chain"
    kind_lower = str(span_kind).lower()
    mapping = {
        "llm": "llm",
        "chain": "chain",
        "agent": "chain",
        "tool": "tool",
        "retriever": "retriever",
        "embedding": "llm",
        "reranker": "chain",
        "guardrail": "chain",
    }
    return mapping.get(kind_lower, "chain")


def _parse_value(value, default_key: str) -> dict:
    """Parse a value into a dict, handling JSON strings, dicts, and lists."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            elif isinstance(parsed, list):
                return {default_key: parsed}
            else:
                return {default_key: parsed}
        except json.JSONDecodeError:
            return {default_key: value}
    elif isinstance(value, dict):
        return value
    elif isinstance(value, list):
        return {default_key: value}
    else:
        return {default_key: str(value)}


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
        # Ensure seg has timestamp prefix - use current time if missing
        if not seg or seg == run_id:
            seg = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + "000000Z" + run_id
        dotted = seg if not parent_dotted else f"{parent_dotted}.{seg}"
        run["dotted_order"] = dotted
        for kid in cmap.get(run_id, []):
            assign(kid, dotted)
    
    assign(root_id, None)
    
    # Ensure ALL runs have dotted_order (catch any orphans)
    for run in runs:
        if not run.get("dotted_order"):
            ts = run.get('start_time') or run.get('end_time')
            seg = _compact_ts(ts) + run["id"]
            if not seg or seg == run["id"]:
                seg = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + "000000Z" + run["id"]
            run["dotted_order"] = seg


def _get_attr(obj, name, default=None):
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# Note: This does not preserve the full original object - it strips out the messages specifically.
# LangSmith does accept a variety of message formats that include additional metadata - this conversion is for simplicity.
def map_phoenix_traces_to_langsmith(traces_dict: dict) -> list[dict]:
    """Transform Phoenix traces (grouped by trace_id) to LangSmith runs format.
    Note that there
    
    Args:
        traces_dict: dict[trace_id -> list of span dicts]
        
    Phoenix DataFrame columns:
    - context.trace_id, context.span_id, parent_id
    - name, span_kind, start_time, end_time
    - attributes.input.value, attributes.output.value
    - attributes.llm.model_name, attributes.openinference.span.kind
    """
    if not traces_dict:
        return []
    
    runs = []
    
    for orig_trace_id, trace_spans in traces_dict.items():
        # Create new IDs for LangSmith
        span_id_mapping = {}
        for span in trace_spans:
            orig_span_id = str(span.get('context.span_id', '') or '')
            if orig_span_id:
                span_id_mapping[orig_span_id] = str(uuid.uuid4())
        
        root_run_id = None
        trace_runs = []
        
        for span in trace_spans:
            orig_span_id = str(span.get('context.span_id', '') or '')
            run_id = span_id_mapping.get(orig_span_id, str(uuid.uuid4()))
            
            orig_parent_id = str(span.get('parent_id', '') or '')
            parent_run_id = span_id_mapping.get(orig_parent_id) if orig_parent_id else None
            
            # First span without parent is the root
            if parent_run_id is None and root_run_id is None:
                root_run_id = run_id
            
            # Get span kind from openinference attribute or span_kind column
            span_kind = str(span.get('attributes.openinference.span.kind', '') or span.get('span_kind', '') or '')
            run_type = _span_kind_to_run_type(span_kind)
            
            # Parse inputs - check multiple sources
            inputs = {}
            tool_params = span.get('attributes.tool.parameters')
            input_value = span.get('attributes.input.value')
            llm_input_messages = span.get('attributes.llm.input_messages')
            
            if tool_params:
                # Tool spans - use parameters
                inputs = _parse_value(tool_params, "parameters")
            elif input_value:
                # General input value (may contain messages for playground)
                inputs = _parse_value(input_value, "input")
            elif llm_input_messages:
                # LLM input messages fallback
                inputs = _parse_value(llm_input_messages, "messages")
            
            # Parse outputs - check multiple sources
            outputs = {}
            output_value = span.get('attributes.output.value')
            llm_output_messages = span.get('attributes.llm.output_messages')
            
            if output_value:
                outputs = _parse_value(output_value, "output")
            elif llm_output_messages:
                outputs = _parse_value(llm_output_messages, "messages")
            
            # Build metadata from attributes
            metadata = {}
            model_name = span.get('attributes.llm.model_name')
            if model_name and isinstance(model_name, str):
                metadata["ls_model_name"] = model_name
            
            run = {
                "id": run_id,
                "trace_id": root_run_id or run_id,
                "name": span.get('name') or "span",
                "run_type": run_type,
                "parent_run_id": parent_run_id,
                "inputs": inputs,
                "outputs": outputs,
                "start_time": safe_isoformat(span.get('start_time')),
                "end_time": safe_isoformat(span.get('end_time')),
                "metadata": metadata,
                "tags": [],
            }
            trace_runs.append(run)
        
        # Fix trace_id for all runs in this trace
        if root_run_id:
            for run in trace_runs:
                run["trace_id"] = root_run_id
            _ensure_end_times(trace_runs)
            _assign_dotted_order(trace_runs, root_run_id)
        
        runs.extend(trace_runs)
    
    return runs


def migrate_traces(workspace_id: str, project_name: str):
    print(f"    - migrating traces...")
    
    total_traces_fetched = 0
    total_spans_fetched = 0
    total_runs_uploaded = 0
    failed_fetch = 0
    failed_transform = 0
    
    try:
        limit = NUM_TRACES_TO_REPLAY if NUM_TRACES_TO_REPLAY and NUM_TRACES_TO_REPLAY > 0 else 100
        traces_dict = phoenix_get_traces(project_name=project_name, limit=limit)
    except Exception as e:
        print(f"       x failed to fetch traces from Phoenix: {e}")
        traceback.print_exc()
        failed_fetch = 1
        print(f"        • Processed traces: 0")
        print(f"        • Failed fetching: {failed_fetch}")
        print(f"        • Failed transforming: {failed_transform}")
        return
    
    if not traces_dict:
        print("       • no traces found")
        print(f"        • Processed traces: 0")
        print(f"        • Failed fetching: {failed_fetch}")
        print(f"        • Failed transforming: {failed_transform}")
        return
    
    total_traces_fetched = len(traces_dict)
    total_spans_fetched = sum(len(spans) for spans in traces_dict.values())
    print(f"       • fetched {total_traces_fetched} traces ({total_spans_fetched} spans) from Phoenix")
    
    try:
        runs = map_phoenix_traces_to_langsmith(traces_dict)
        if not runs:
            print("       • no runs to upload after transformation")
            failed_transform = total_traces_fetched
        else:
            ls_replay_runs_sdk(workspace_id, runs, project_name=project_name)
            total_runs_uploaded = len(runs)
            unique_traces = len(set(r["trace_id"] for r in runs))
            print(f"       • uploaded {total_runs_uploaded} spans ({unique_traces} traces) to project '{project_name}'")
    except Exception as e:
        print(f"       x failed to transform/upload traces: {e}")
        traceback.print_exc()
        failed_transform = total_traces_fetched
    
    print(f"        • Processed traces: {total_traces_fetched}")
    print(f"        • Failed fetching: {failed_fetch}")
    print(f"        • Failed transforming: {failed_transform}")
