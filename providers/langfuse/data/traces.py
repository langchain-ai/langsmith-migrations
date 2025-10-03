import os
import sys
import uuid
import datetime as dt
import time
import json
from types import SimpleNamespace
# Corrected location for MapValue:
from langfuse.api.resources.commons.types import MapValue
# Ingestion types:
from langfuse.api.resources.ingestion.types import (
    TraceBody,
    CreateSpanBody,
    CreateGenerationBody,
    CreateEventBody,
    ScoreBody,
    IngestionEvent_TraceCreate,
    IngestionEvent_SpanCreate,
    IngestionEvent_GenerationCreate,
    IngestionEvent_EventCreate,
    IngestionEvent_ScoreCreate,
    IngestionUsage,
)
# Other common types:
from langfuse.api.resources.commons.types import ObservationLevel, ScoreSource, Usage
from langfuse.api.resources.commons.types.score import Score_Numeric, Score_Categorical, Score_Boolean
from dotenv import load_dotenv
from config import NUM_TRACES_TO_REPLAY
from utils.langsmith import ls_upload_runs
from utils.langfuse import lf_get
 
load_dotenv()

# --- Helper Function for Robust Datetime Formatting ---
def safe_isoformat(dt_obj):
    """Safely formats datetime object to ISO 8601 string, handling None."""
    if dt_obj is None:
        return None
    if not isinstance(dt_obj, dt.datetime):
        if isinstance(dt_obj, str): # Allow pre-formatted strings
             try:
                 dt.datetime.fromisoformat(dt_obj.replace('Z', '+00:00'))
                 return dt_obj
             except ValueError:
                 return None
        return None
    try:
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        iso_str = dt_obj.isoformat(timespec='milliseconds')
        if iso_str.endswith('+00:00'):
            iso_str = iso_str[:-6] + 'Z'
        return iso_str
    except Exception:
        return None


def map_langfuse_to_langsmith(source_trace):
    """
    Maps Langfuse trace data to LangSmith-compatible format.
    Returns a list of LangSmith run objects.
    """
    langsmith_runs = []
    
    # Map trace to LangSmith run
    trace_run = {
        "id": source_trace.id,
        "name": getattr(source_trace, 'name', None) or "Trace",
        "run_type": "chain",
        "session_id": getattr(source_trace, 'session_id', None),
        "session_name": None,
        "tags": getattr(source_trace, 'tags', []) or [],
        "metadata": source_trace.metadata if isinstance(getattr(source_trace, 'metadata', {}), dict) else {},
        "inputs": getattr(source_trace, 'input', None),
        "outputs": getattr(source_trace, 'output', None),
        "start_time": safe_isoformat(getattr(source_trace, 'timestamp', None)),
        "end_time": None,
        "status": "completed",
        "error": None,
        "invocation_params": {},
        "usage_metadata": {},
        "child_runs": []
    }
    
    # Process observations to create child runs
    observations = getattr(source_trace, 'observations', []) or []
    sorted_observations = sorted(observations, key=lambda o: getattr(o, 'start_time', None) or '')
    observation_runs = {}
    
    for obs in sorted_observations:
        run_id = str(uuid.uuid4())
        observation_runs[getattr(obs, 'id', run_id)] = run_id
        
        # Determine run type based on observation type
        run_type = "chain"
        if getattr(obs, 'type', '').upper() == "GENERATION":
            run_type = "llm"
        elif getattr(obs, 'type', '').upper() == "EVENT":
            run_type = "tool"
        
        # Map model information
        invocation_params = {}
        if getattr(obs, 'model', None):
            invocation_params["model"] = obs.model
        if isinstance(getattr(obs, 'model_parameters', None), dict):
            param_mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "max_tokens": "max_tokens",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
                "seed": "seed",
                "stop": "stop_sequences",
                "top_k": "top_k"
            }
            for langfuse_key, langsmith_key in param_mapping.items():
                if langfuse_key in obs.model_parameters:
                    invocation_params[langsmith_key] = obs.model_parameters[langfuse_key]
        
        # Map usage information
        usage_metadata = {}
        usage = getattr(obs, 'usage', None)
        if isinstance(usage, dict):
            if usage.get('input') is not None:
                usage_metadata['input_tokens'] = usage.get('input')
            if usage.get('output') is not None:
                usage_metadata['output_tokens'] = usage.get('output')
            if usage.get('total') is not None:
                usage_metadata['total_tokens'] = usage.get('total')
        
        inputs = getattr(obs, 'input', None)
        outputs = getattr(obs, 'output', None)
        
        if getattr(obs, 'type', '').upper() == "GENERATION" and getattr(obs, 'model', None):
            if isinstance(inputs, dict) and "messages" in inputs:
                inputs = inputs
            elif isinstance(inputs, str):
                inputs = {"prompt": inputs}
            if isinstance(outputs, dict) and "messages" in outputs:
                outputs = outputs
            elif isinstance(outputs, str):
                outputs = {"completion": outputs}
        
        run = {
            "id": run_id,
            "name": getattr(obs, 'name', None) or f"{str(getattr(obs, 'type', 'obs')).lower()}_{getattr(obs, 'id', '')}",
            "run_type": run_type,
            "parent_run_id": observation_runs.get(getattr(obs, 'parent_observation_id', None)),
            "session_id": getattr(source_trace, 'session_id', None),
            "tags": [],
            "metadata": obs.metadata if isinstance(getattr(obs, 'metadata', {}), dict) else {},
            "inputs": inputs,
            "outputs": outputs,
            "start_time": safe_isoformat(getattr(obs, 'start_time', None)),
            "end_time": safe_isoformat(getattr(obs, 'end_time', None)) if getattr(obs, 'end_time', None) else None,
            "status": "completed",
            "error": getattr(obs, 'status_message', None) or None,
            "invocation_params": invocation_params,
            "usage_metadata": usage_metadata
        }
        
        if getattr(obs, 'type', '').upper() == "GENERATION" and getattr(obs, 'model', None):
            model_lower = obs.model.lower()
            if "openai" in model_lower:
                run["metadata"]["ls_provider"] = "openai"
            elif "anthropic" in model_lower:
                run["metadata"]["ls_provider"] = "anthropic"
            elif "google" in model_lower:
                run["metadata"]["ls_provider"] = "google"
            else:
                run["metadata"]["ls_provider"] = "unknown"
        
        langsmith_runs.append(run)
    
    # Add scores as feedback
    scores = getattr(source_trace, 'scores', []) or []
    for score in scores:
        score_obs_id = getattr(score, 'observation_id', None)
        feedback = {
            "id": str(uuid.uuid4()),
            "run_id": observation_runs.get(score_obs_id, source_trace.id),
            "key": getattr(score, 'name', None),
            "score": getattr(score, 'value', None),
            "comment": getattr(score, 'comment', None),
            "metadata": score.metadata if isinstance(getattr(score, 'metadata', {}), dict) else {},
            "source": getattr(score, 'source', None),
            "timestamp": safe_isoformat(getattr(score, 'timestamp', None))
        }
        langsmith_runs.append({"type": "feedback", **feedback})
    
    return langsmith_runs


def _wrap(obj):
    """Recursively wrap dicts into SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [ _wrap(v) for v in obj ]
    return obj


def transform_trace_to_ingestion_batch(source_trace):
    """
    Transforms a fetched TraceWithFullDetails object into a list of
    IngestionEvent objects suitable for the batch ingestion endpoint.
    Uses the ORIGINAL source trace ID for the new trace.
    Generates new IDs for observations/scores within the trace.
    Maps parent/child relationships using new observation IDs.
    """
    ingestion_events = []
    preserved_trace_id = source_trace.id
    obs_id_map = {}
 
    # 1. Create Trace Event
    trace_metadata = source_trace.metadata if isinstance(source_trace.metadata, dict) else {}
    trace_body = TraceBody(
        id=preserved_trace_id,
        timestamp=source_trace.timestamp,
        name=source_trace.name,
        user_id=source_trace.user_id,
        input=source_trace.input,
        output=source_trace.output,
        session_id=source_trace.session_id,
        release=source_trace.release,
        version=source_trace.version,
        metadata=trace_metadata or None,
        tags=source_trace.tags if source_trace.tags is not None else [],
        public=source_trace.public,
        environment=source_trace.environment if source_trace.environment else "default",
    )
    event_timestamp_str = safe_isoformat(dt.datetime.now(dt.timezone.utc))
    if not event_timestamp_str:
         print("Error: Could not format timestamp for trace event. Skipping trace.")
         return []
    trace_event_id = str(uuid.uuid4())
    ingestion_events.append(
        IngestionEvent_TraceCreate(id=trace_event_id, timestamp=event_timestamp_str, body=trace_body)
    )
 
    # 2. Create Observation Events
    sorted_observations = sorted(source_trace.observations, key=lambda o: o.start_time)
    for source_obs in sorted_observations:
        new_obs_id = str(uuid.uuid4())
        obs_id_map[source_obs.id] = new_obs_id
        new_parent_observation_id = obs_id_map.get(source_obs.parent_observation_id) if source_obs.parent_observation_id else None
        obs_metadata = source_obs.metadata if isinstance(source_obs.metadata, dict) else {}
 
        model_params_mapped = None
        if isinstance(source_obs.model_parameters, dict): model_params_mapped = source_obs.model_parameters
        elif source_obs.model_parameters is not None: print(f"Warning: Obs {source_obs.id} model_parameters type {type(source_obs.model_parameters)}, skipping.")
 
        common_body_args = {
            "id": new_obs_id, "trace_id": preserved_trace_id, "name": source_obs.name,
            "start_time": source_obs.start_time, "metadata": obs_metadata or None,
            "input": source_obs.input, "output": source_obs.output, "level": source_obs.level,
            "status_message": source_obs.status_message, "parent_observation_id": new_parent_observation_id,
            "version": source_obs.version, "environment": source_obs.environment if source_obs.environment else "default",
        }
 
        event_body = None; ingestion_event_type = None
        event_specific_timestamp = safe_isoformat(dt.datetime.now(dt.timezone.utc))
        if not event_specific_timestamp: print(f"Error: Could not format timestamp for obs {new_obs_id}. Skipping."); continue
 
        try:
            if source_obs.type == "SPAN":
                event_body = CreateSpanBody(**common_body_args, end_time=source_obs.end_time)
                ingestion_event_type = IngestionEvent_SpanCreate
            elif source_obs.type == "EVENT":
                event_body = CreateEventBody(**common_body_args)
                ingestion_event_type = IngestionEvent_EventCreate
            elif source_obs.type == "GENERATION":
                usage_to_pass = None
                if isinstance(source_obs.usage, Usage):
                    usage_data = {k: getattr(source_obs.usage, k, None) for k in ['input', 'output', 'total', 'unit', 'input_cost', 'output_cost', 'total_cost']}
                    filtered_usage_data = {k: v for k, v in usage_data.items() if v is not None}
                    if filtered_usage_data: usage_to_pass = Usage(**filtered_usage_data)
                elif source_obs.usage is not None: print(f"Warning: Obs {source_obs.id} has usage type {type(source_obs.usage)}. Skipping.")
 
                event_body = CreateGenerationBody(
                    **common_body_args, end_time=source_obs.end_time,
                    completion_start_time=source_obs.completion_start_time,
                    model=source_obs.model, model_parameters=model_params_mapped,
                    usage=usage_to_pass, cost_details=source_obs.cost_details,
                    usage_details=source_obs.usage_details,
                    prompt_name=getattr(source_obs, 'prompt_name', None),
                    prompt_version=getattr(source_obs, 'prompt_version', None),
                )
                ingestion_event_type = IngestionEvent_GenerationCreate
            else: print(f"Warning: Unknown obs type '{source_obs.type}' for ID {source_obs.id}. Skipping."); continue
 
            if event_body and ingestion_event_type:
                event_envelope_id = str(uuid.uuid4())
                ingestion_events.append(
                    ingestion_event_type(id=event_envelope_id, timestamp=event_specific_timestamp, body=event_body)
                )
        except Exception as e: print(f"Error creating obs body for {source_obs.id} (type: {source_obs.type}): {e}"); continue
 
    # 3. Create Score Events
    for source_score in source_trace.scores:
        new_score_id = str(uuid.uuid4())
        new_observation_id = obs_id_map.get(source_score.observation_id) if source_score.observation_id else None
        score_metadata = source_score.metadata if isinstance(source_score.metadata, dict) else {}
 
        score_body_value = None
        if source_score.data_type == "CATEGORICAL":
            # For categorical, use the string_value field from the source
             if hasattr(source_score, 'string_value') and isinstance(getattr(source_score, 'string_value', None), str):
                 score_body_value = source_score.string_value
             else:
                 # Fallback or warning if string_value is missing for categorical
                 print(f"      Warning: Categorical score {source_score.id} is missing string_value. Attempting to use numeric value '{source_score.value}' as string.")
                 score_body_value = str(source_score.value) if source_score.value is not None else None
 
        elif source_score.data_type in ["NUMERIC", "BOOLEAN"]:
            # For numeric/boolean, use the numeric value field
            score_body_value = source_score.value # Already float or None
        else:
            print(f"      Warning: Unknown score dataType '{source_score.data_type}' for score {source_score.id}. Attempting numeric value.")
            score_body_value = source_score.value
 
        # If after all checks, value is still None, skip score
        if score_body_value is None:
             print(f"      Warning: Could not determine valid value for score {source_score.id} (dataType: {source_score.data_type}). Skipping score.")
             continue
 
        try:
            score_body = ScoreBody(
                id=new_score_id,
                trace_id=preserved_trace_id,
                name=source_score.name,
                # Pass the correctly typed value
                value=score_body_value,
                # string_value field might not be needed if value holds the category string
                # string_value=string_value if source_score.data_type == "CATEGORICAL" else None, # Optional: maybe pass string_value only for categorical?
                source=source_score.source,
                comment=source_score.comment,
                observation_id=new_observation_id,
                timestamp=source_score.timestamp,
                config_id=source_score.config_id,
                metadata=score_metadata or None,
                data_type=source_score.data_type,
                environment=source_score.environment if source_score.environment else "default",
            )
            event_timestamp_str = safe_isoformat(dt.datetime.now(dt.timezone.utc))
            if not event_timestamp_str: print(f"Error: Could not format timestamp for score {new_score_id}. Skipping."); continue
            event_envelope_id = str(uuid.uuid4())
            ingestion_events.append(
                IngestionEvent_ScoreCreate(id=event_envelope_id, timestamp=event_timestamp_str, body=score_body)
            )
        except Exception as e: print(f"Error creating score body for {source_score.id}: {e}"); continue
 
    return ingestion_events
 
 
def fetch_and_transform_traces(workspace_id: str, sleep_between_gets=0.7, max_retries=4):
    """
    Fetch most recent traces using Public API and transform them into ingestion events.
    Enforces NUM_TRACES_TO_REPLAY as a hard cap.
    """
    try:
        mt = int(NUM_TRACES_TO_REPLAY)
        max_traces = mt if mt > 0 else None
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
                source_obj = _wrap(source_detail)
                runs_batch = map_langfuse_to_langsmith(source_obj)
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

    # Upload accumulated runs to LangSmith workspace
    if accumulated_runs:
        try:
            ls_upload_runs(workspace_id, accumulated_runs)
        except Exception as e:
            print(f"Error uploading runs to LangSmith: {e}")

    print(f"        • Processed traces: {total_processed}")
    print(f"        • Failed fetching details (after retries): {total_failed_fetch}")
    print(f"        • Failed transforming data (incl. skipping): {total_failed_transform}")

def migrate_traces(workspace_id: str, project_id: str):
    print(f"    - migrating recent traces…")
    fetch_and_transform_traces(workspace_id)