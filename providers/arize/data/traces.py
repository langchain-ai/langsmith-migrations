from __future__ import annotations

import traceback
import uuid
import datetime as dt
from config import NUM_TRACES_TO_REPLAY
from utils.arize import arize_export_traces
from utils.langsmith import ls_replay_runs_sdk


def safe_isoformat(dt_obj):
    if dt_obj is None:
        return None
    if not isinstance(dt_obj, dt.datetime):
        if isinstance(dt_obj, str):
            return dt_obj
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


def _get_col(row, *col_names, default=None):
    import pandas as pd
    import numpy as np
    for col in col_names:
        if col in row.index:
            val = row[col]
            if val is None:
                continue
            if isinstance(val, (list, np.ndarray)):
                if len(val) == 0:
                    continue
                return val
            try:
                if pd.isna(val):
                    continue
            except (ValueError, TypeError):
                pass
            if isinstance(val, float) and str(val) == 'nan':
                continue
            if isinstance(val, str) and val.strip() == '':
                continue
            return val
    return default


def map_arize_spans_to_langsmith(traces_df) -> list[dict]:
    """Transform Arize trace DataFrame to LangSmith runs format.
    
    Note: The following Arize data is not explicitly migrated, but may be preserved in metadata:
    - Span events and exceptions (only status is preserved)
    - Evaluation scores and annotations
    - Session/conversation groupings
    - Cost tracking data
    - Retrieval document contents (only query is preserved)
    """
    if traces_df is None or traces_df.empty:
        return []
    
    runs = []
    
    trace_id_col = None
    for col in ['context.trace_id', 'trace_id', 'context_trace_id']:
        if col in traces_df.columns:
            trace_id_col = col
            break
    
    if not trace_id_col:
        print("      ! Could not find trace_id column in Arize export")
        return []
    
    span_id_mapping = {}
    
    for trace_id, trace_group in traces_df.groupby(trace_id_col):
        new_trace_id = str(uuid.uuid4())
        
        trace_spans = trace_group.sort_values(
            by=[c for c in ['start_time', 'context.span_id', 'span_id'] if c in trace_group.columns][:1]
        )
        
        for _, row in trace_spans.iterrows():
            orig_span_id = _get_col(row, 'context.span_id', 'span_id', 'context_span_id')
            if orig_span_id:
                span_id_mapping[str(orig_span_id)] = str(uuid.uuid4())
        
        root_run_id = None
        for _, row in trace_spans.iterrows():
            orig_span_id = _get_col(row, 'context.span_id', 'span_id', 'context_span_id')
            run_id = span_id_mapping.get(str(orig_span_id), str(uuid.uuid4()))
            
            orig_parent_id = _get_col(row, 'parent_id', 'parent_span_id')
            parent_run_id = span_id_mapping.get(str(orig_parent_id)) if orig_parent_id else None
            
            if parent_run_id is None and root_run_id is None:
                root_run_id = run_id
            
            span_kind = _get_col(row, 'span_kind', 'openinference.span.kind', 'attributes.openinference.span.kind')
            run_type = _span_kind_to_run_type(span_kind)
            span_name = _get_col(row, 'name', 'span_name') or "span"
            
            inputs = {}
            
            input_value = _get_col(row, 'attributes.input.value', 'input.value')
            if input_value:
                inputs["input"] = input_value
            
            llm_input = _get_col(row, 'attributes.llm.input_messages', 'llm.input_messages')
            if llm_input:
                inputs["messages"] = llm_input
            llm_prompt = _get_col(row, 'attributes.llm.prompt')
            if llm_prompt and "input" not in inputs:
                inputs["input"] = llm_prompt
            
            tool_params = _get_col(row, 'attributes.tool.parameters', 'attributes.tool.arguments')
            if tool_params:
                inputs["tool_parameters"] = tool_params
            tool_name = _get_col(row, 'attributes.tool.name')
            if tool_name:
                inputs["tool_name"] = tool_name
            
            retrieval_query = _get_col(row, 'attributes.retrieval.query')
            if retrieval_query:
                inputs["query"] = retrieval_query
            
            outputs = {}
            
            output_value = _get_col(row, 'attributes.output.value', 'output.value')
            if output_value:
                outputs["output"] = output_value
            
            llm_output = _get_col(row, 'attributes.llm.output_messages', 'llm.output_messages')
            if llm_output:
                outputs["messages"] = llm_output
            llm_response = _get_col(row, 'attributes.llm.response', 'attributes.output.response')
            if llm_response and "output" not in outputs:
                outputs["output"] = llm_response
            
            tool_result = _get_col(row, 'attributes.tool.result')
            if tool_result:
                outputs["tool_result"] = tool_result
            
            metadata = {}
            
            model_name = _get_col(row, 'attributes.llm.model_name', 'attributes.llm.model')
            if model_name:
                metadata["ls_model_name"] = model_name
                model_lower = str(model_name).lower()
                if "gpt" in model_lower or "openai" in model_lower:
                    metadata["ls_provider"] = "openai"
                elif "claude" in model_lower or "anthropic" in model_lower:
                    metadata["ls_provider"] = "anthropic"
            
            prompt_tokens = _get_col(row, 'attributes.llm.token_count.prompt')
            completion_tokens = _get_col(row, 'attributes.llm.token_count.completion')
            total_tokens = _get_col(row, 'attributes.llm.token_count.total')
            if prompt_tokens or completion_tokens or total_tokens:
                metadata["token_usage"] = {}
                if prompt_tokens:
                    metadata["token_usage"]["prompt_tokens"] = int(prompt_tokens)
                if completion_tokens:
                    metadata["token_usage"]["completion_tokens"] = int(completion_tokens)
                if total_tokens:
                    metadata["token_usage"]["total_tokens"] = int(total_tokens)
            
            invocation_params = _get_col(row, 'attributes.llm.invocation_parameters')
            if invocation_params:
                metadata["invocation_params"] = invocation_params
            
            for col in row.index:
                if col.startswith('attributes.metadata.') or col.startswith('metadata.'):
                    val = _get_col(row, col)
                    if val is not None:
                        key = col.replace('attributes.metadata.', '').replace('metadata.', '')
                        metadata[key] = val
            
            start_time = _get_col(row, 'start_time')
            end_time = _get_col(row, 'end_time')
            
            trace_id = run_id if parent_run_id is None else root_run_id
            
            run = {
                "id": run_id,
                "trace_id": trace_id,
                "name": _get_col(row, 'name', 'span_name') or "span",
                "run_type": run_type,
                "parent_run_id": parent_run_id,
                "inputs": inputs if inputs else {},
                "outputs": outputs if outputs else {},
                "start_time": safe_isoformat(start_time) if start_time else None,
                "end_time": safe_isoformat(end_time) if end_time else None,
                "metadata": metadata,
                "tags": [],
            }
            runs.append(run)
        
        if root_run_id:
            trace_runs = [r for r in runs if r["trace_id"] == root_run_id]
            _ensure_end_times(trace_runs)
            _assign_dotted_order(trace_runs, root_run_id)
    
    return runs


def migrate_traces(workspace_id: str, project_name: str, days_back: int = 7):
    print(f"    - migrating traces (last {days_back} days)...")
    
    try:
        if NUM_TRACES_TO_REPLAY and NUM_TRACES_TO_REPLAY > 0:
            days_back = max(1, NUM_TRACES_TO_REPLAY // 100) or days_back
        
        traces_df = arize_export_traces(project_name, days_back=days_back)
    except Exception as e:
        print(f"       x failed to export traces from Arize: {e}")
        traceback.print_exc()
        return
    
    if traces_df is None or traces_df.empty:
        print("       • no traces found")
        return
    
    print(f"       • exported {len(traces_df)} spans from Arize")
    
    try:
        runs = map_arize_spans_to_langsmith(traces_df)
        if not runs:
            print("       • no runs to upload after transformation")
            return
        
        if NUM_TRACES_TO_REPLAY and NUM_TRACES_TO_REPLAY > 0:
            unique_traces = set(r["trace_id"] for r in runs)
            if len(unique_traces) > NUM_TRACES_TO_REPLAY:
                trace_ids = list(unique_traces)[:NUM_TRACES_TO_REPLAY]
                runs = [r for r in runs if r["trace_id"] in trace_ids]
        
        ls_replay_runs_sdk(workspace_id, runs, project_name=project_name)
        unique_traces = len(set(r["trace_id"] for r in runs))
        print(f"       • uploaded {len(runs)} spans ({unique_traces} traces) to project '{project_name}'")
    except Exception as e:
        print(f"       x failed to transform/upload traces: {e}")
        traceback.print_exc()
