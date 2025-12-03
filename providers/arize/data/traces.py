from __future__ import annotations

import traceback
import uuid
import datetime as dt
from config import NUM_TRACES_TO_REPLAY, ARIZE_PROJECT_NAME
from utils.arize import arize_export_traces
from utils.langsmith import ls_replay_runs_sdk


def safe_isoformat(dt_obj):
    """Convert datetime to ISO format string."""
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
    """Create compact timestamp for dotted_order."""
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
    """Map OpenInference span kind to LangSmith run type."""
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
    """Ensure all runs have end_time set."""
    for r in runs:
        if isinstance(r, dict) and r.get("end_time") is None:
            r["end_time"] = r.get("start_time")


def _children_map(runs: list[dict]) -> dict:
    """Build parent->children mapping."""
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
    """Assign dotted_order for LangSmith trace hierarchy."""
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
    """Get value from row trying multiple column names."""
    import pandas as pd
    import numpy as np
    for col in col_names:
        if col in row.index:
            val = row[col]
            # Handle various null types
            if val is None:
                continue
            # Handle arrays/lists (don't check isna on these)
            if isinstance(val, (list, np.ndarray)):
                if len(val) == 0:
                    continue
                return val
            # Handle scalar null checks
            try:
                if pd.isna(val):
                    continue
            except (ValueError, TypeError):
                # pd.isna fails on some types, assume not null
                pass
            if isinstance(val, float) and str(val) == 'nan':
                continue
            if isinstance(val, str) and val.strip() == '':
                continue
            return val
    return default


def map_arize_spans_to_langsmith(traces_df) -> list[dict]:
    """Transform Arize trace DataFrame to LangSmith runs format.
    
    Arize exports spans in OpenInference format with columns like:
    - context.trace_id, context.span_id
    - parent_id
    - name
    - span_kind (LLM, CHAIN, TOOL, etc.)
    - start_time, end_time
    - attributes.input.value, attributes.output.value
    - attributes.llm.input_messages, attributes.llm.output_messages
    - attributes.metadata.*
    """
    if traces_df is None or traces_df.empty:
        return []
    
    runs = []
    
    # Group spans by trace_id
    trace_id_col = None
    for col in ['context.trace_id', 'trace_id', 'context_trace_id']:
        if col in traces_df.columns:
            trace_id_col = col
            break
    
    if not trace_id_col:
        print("      ! Could not find trace_id column in Arize export")
        return []
    
    # Create new UUIDs for LangSmith (they need valid UUIDs)
    span_id_mapping = {}  # original span_id -> new UUID
    
    for trace_id, trace_group in traces_df.groupby(trace_id_col):
        # Generate new trace UUID for LangSmith
        new_trace_id = str(uuid.uuid4())
        
        # Sort spans by start_time
        trace_spans = trace_group.sort_values(
            by=[c for c in ['start_time', 'context.span_id', 'span_id'] if c in trace_group.columns][:1]
        )
        
        # First pass: create ID mapping
        for _, row in trace_spans.iterrows():
            orig_span_id = _get_col(row, 'context.span_id', 'span_id', 'context_span_id')
            if orig_span_id:
                span_id_mapping[str(orig_span_id)] = str(uuid.uuid4())
        
        # Second pass: create runs
        root_run_id = None
        for _, row in trace_spans.iterrows():
            orig_span_id = _get_col(row, 'context.span_id', 'span_id', 'context_span_id')
            run_id = span_id_mapping.get(str(orig_span_id), str(uuid.uuid4()))
            
            # Get parent
            orig_parent_id = _get_col(row, 'parent_id', 'parent_span_id')
            parent_run_id = span_id_mapping.get(str(orig_parent_id)) if orig_parent_id else None
            
            # Track root
            if parent_run_id is None and root_run_id is None:
                root_run_id = run_id
            
            # Get span kind and map to run type
            span_kind = _get_col(row, 'span_kind', 'openinference.span.kind', 'attributes.openinference.span.kind')
            run_type = _span_kind_to_run_type(span_kind)
            span_name = _get_col(row, 'name', 'span_name') or "span"
            
            # Extract inputs based on span type
            inputs = {}
            
            # General input
            input_value = _get_col(row, 'attributes.input.value', 'input.value')
            if input_value:
                inputs["input"] = input_value
            
            # LLM inputs
            llm_input = _get_col(row, 'attributes.llm.input_messages', 'llm.input_messages')
            if llm_input:
                inputs["messages"] = llm_input
            llm_prompt = _get_col(row, 'attributes.llm.prompt')
            if llm_prompt and "input" not in inputs:
                inputs["input"] = llm_prompt
            
            # Tool inputs
            tool_params = _get_col(row, 'attributes.tool.parameters', 'attributes.tool.arguments')
            if tool_params:
                inputs["tool_parameters"] = tool_params
            tool_name = _get_col(row, 'attributes.tool.name')
            if tool_name:
                inputs["tool_name"] = tool_name
            
            # Retriever inputs
            retrieval_query = _get_col(row, 'attributes.retrieval.query')
            if retrieval_query:
                inputs["query"] = retrieval_query
            
            # Extract outputs based on span type
            outputs = {}
            
            # General output
            output_value = _get_col(row, 'attributes.output.value', 'output.value')
            if output_value:
                outputs["output"] = output_value
            
            # LLM outputs
            llm_output = _get_col(row, 'attributes.llm.output_messages', 'llm.output_messages')
            if llm_output:
                outputs["messages"] = llm_output
            llm_response = _get_col(row, 'attributes.llm.response', 'attributes.output.response')
            if llm_response and "output" not in outputs:
                outputs["output"] = llm_response
            
            # Tool outputs
            tool_result = _get_col(row, 'attributes.tool.result')
            if tool_result:
                outputs["tool_result"] = tool_result
            
            # Extract metadata
            metadata = {}
            
            # Model info
            model_name = _get_col(row, 'attributes.llm.model_name', 'attributes.llm.model')
            if model_name:
                metadata["ls_model_name"] = model_name
                # Set provider hint for LangSmith
                model_lower = str(model_name).lower()
                if "gpt" in model_lower or "openai" in model_lower:
                    metadata["ls_provider"] = "openai"
                elif "claude" in model_lower or "anthropic" in model_lower:
                    metadata["ls_provider"] = "anthropic"
            
            # Token counts
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
            
            # Invocation params
            invocation_params = _get_col(row, 'attributes.llm.invocation_parameters')
            if invocation_params:
                metadata["invocation_params"] = invocation_params
            
            # Get timestamps
            start_time = _get_col(row, 'start_time')
            end_time = _get_col(row, 'end_time')
            
            # For root runs, trace_id should equal run_id
            # For child runs, trace_id should equal root_run_id
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
        
        # Assign dotted order for this trace
        if root_run_id:
            trace_runs = [r for r in runs if r["trace_id"] == root_run_id]
            _ensure_end_times(trace_runs)
            _assign_dotted_order(trace_runs, root_run_id)
    
    return runs


def migrate_traces(workspace_id: str, days_back: int = 7):
    """Migrate traces from Arize to LangSmith.
    
    Args:
        workspace_id: LangSmith workspace ID.
        days_back: Number of days of traces to export (default: 7).
    """
    print(f"    - migrating traces (last {days_back} days)...")
    
    try:
        # Use NUM_TRACES_TO_REPLAY as a hint for days if set
        if NUM_TRACES_TO_REPLAY and NUM_TRACES_TO_REPLAY > 0:
            # Rough heuristic: assume ~100 traces per day
            days_back = max(1, NUM_TRACES_TO_REPLAY // 100) or days_back
        
        traces_df = arize_export_traces(days_back=days_back)
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
        
        # Limit if NUM_TRACES_TO_REPLAY is set
        if NUM_TRACES_TO_REPLAY and NUM_TRACES_TO_REPLAY > 0:
            # Count unique traces
            unique_traces = set(r["trace_id"] for r in runs)
            if len(unique_traces) > NUM_TRACES_TO_REPLAY:
                # Filter to first N traces
                trace_ids = list(unique_traces)[:NUM_TRACES_TO_REPLAY]
                runs = [r for r in runs if r["trace_id"] in trace_ids]
        
        ls_replay_runs_sdk(workspace_id, runs, project_name=ARIZE_PROJECT_NAME)
        unique_traces = len(set(r["trace_id"] for r in runs))
        print(f"       • uploaded {len(runs)} spans ({unique_traces} traces) to project '{ARIZE_PROJECT_NAME}'")
    except Exception as e:
        print(f"       x failed to transform/upload traces: {e}")
        traceback.print_exc()
