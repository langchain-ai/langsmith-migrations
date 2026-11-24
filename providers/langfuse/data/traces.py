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
from utils.langfuse import (
    lf_get,
    lf_get_trace_observations,
    lf_recent_trace_ids,
    lf_supports_observations_v2,
)
 
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

# Note: This does not preserve the full original object - it strips out the messages specifically.
# LangSmith does accept a variety of message formats that include additional metadata - this conversion is for simplicity.
def _to_messages(obj, default_role="user"):
    obj = _to_plain(obj)
    if isinstance(obj, dict) and "message" in obj:
        m = obj["message"]
        if isinstance(m, dict) and {"role", "content"}.issubset(m.keys()):
            return {"messages": [{"role": m.get("role") or default_role, "content": m.get("content")}]}
        if isinstance(m, (list, tuple)) and len(m) == 2:
            role, content = m[0], m[1]
            return {"messages": [{"role": str(role) or default_role, "content": content}]}
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        msgs = []
        for m in obj["messages"]:
            if isinstance(m, dict) and {"role", "content"}.issubset(m.keys()):
                msgs.append({"role": m.get("role") or default_role, "content": m.get("content")})
            elif isinstance(m, str):
                msgs.append({"role": default_role, "content": m})
        if not msgs:
            return obj
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
        return obj
    return obj


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


def _provider_from_model(model: str) -> str:
    m = (model or "").lower()
    if any(x in m for x in ("claude", "anthropic")):
        return "anthropic"
    if any(x in m for x in ("gpt", "openai", "o1", "o3", "o4", "text-embedding")):
        return "openai"
    if any(x in m for x in ("gemini", "google")):
        return "google"
    return "unknown"


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_TOTAL_KEYS = {"total", "totalTokens", "total_tokens"}


def _is_output_key(k: str) -> bool:
    return k.startswith("output") or k.startswith("completion")


def _usage_metadata(obs) -> dict | None:
    """Map a Langfuse generation's token usage and cost to LangSmith usage_metadata.

    Langfuse usage keys are disjoint and add up to "total": cached prompt tokens
    are separate keys (Anthropic: input + cache_read_input_tokens +
    cache_creation_input_tokens). LangSmith's input_tokens counts every prompt
    token, with the cached part broken out in input_token_details.
    """
    usage = {}
    for field in ("usageDetails", "usage_details", "usage"):
        v = _to_plain(_get_attr(obs, [field]))
        if isinstance(v, dict) and any(_is_number(x) for x in v.values()):
            usage = {k: x for k, x in v.items() if _is_number(x)}
            break
    out: dict = {}
    if usage:
        total = next((usage[k] for k in _TOTAL_KEYS if k in usage), None)
        out_tokens = sum(x for k, x in usage.items() if _is_output_key(k))
        in_parts = {k: x for k, x in usage.items() if k not in _TOTAL_KEYS and not _is_output_key(k)}
        in_tokens = sum(in_parts.values())
        if total is not None and total >= out_tokens:
            in_tokens = total - out_tokens
        out["input_tokens"] = in_tokens
        out["output_tokens"] = out_tokens
        out["total_tokens"] = total if total is not None else in_tokens + out_tokens
        cache_read = sum(x for k, x in in_parts.items() if "cache_read" in k or "cached" in k)
        cache_write = sum(x for k, x in in_parts.items() if "cache_creation" in k or "cache_write" in k)
        details = {k: v for k, v in (("cache_read", cache_read), ("cache_creation", cache_write)) if v}
        if details:
            out["input_token_details"] = details
        reasoning = sum(x for k, x in usage.items() if _is_output_key(k) and "reasoning" in k)
        if reasoning:
            out["output_token_details"] = {"reasoning": reasoning}

    # Cost: v3 trace detail has calculated*Cost; v4 observations have costDetails + totalCost.
    cost = _to_plain(_get_attr(obs, ["costDetails", "cost_details"]))
    cost = {k: x for k, x in cost.items() if _is_number(x)} if isinstance(cost, dict) else {}
    total_cost = _get_attr(obs, ["calculatedTotalCost", "totalCost"])
    if not _is_number(total_cost):
        total_cost = cost.get("total")
    output_cost = _get_attr(obs, ["calculatedOutputCost", "outputCost"])
    if not _is_number(output_cost) and cost:
        output_cost = sum(x for k, x in cost.items() if _is_output_key(k))
    input_cost = _get_attr(obs, ["calculatedInputCost", "inputCost"])
    if not _is_number(input_cost) and cost:
        input_cost = sum(x for k, x in cost.items() if k != "total" and not _is_output_key(k))
    for key, val in (("input_cost", input_cost), ("output_cost", output_cost), ("total_cost", total_cost)):
        if _is_number(val):
            out[key] = val
    return out or None


def _error_of(obs) -> str | None:
    """Langfuse marks failures with level ERROR and a status message."""
    if str(_get_attr(obs, ["level"]) or "").upper() == "ERROR":
        return _get_attr(obs, ["statusMessage", "status_message"]) or "error"
    return None


def _parse_ts(val) -> dt.datetime | None:
    if not val:
        return None
    if isinstance(val, dt.datetime):
        return val if val.tzinfo else val.replace(tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _rebase_times(runs: list[dict], new_start: dt.datetime):
    """Shift every run of a trace so the root starts at new_start, keeping durations."""
    root_start = _parse_ts(runs[0].get("start_time"))
    if root_start is None:
        return
    shift = new_start - root_start
    for r in runs:
        for key in ("start_time", "end_time"):
            t = _parse_ts(r.get(key))
            if t is not None:
                r[key] = safe_isoformat(t + shift)


def map_langfuse_to_langsmith(source_trace, rebase_to: dt.datetime | None = None):
    """Convert one Langfuse trace to LangSmith runs.

    rebase_to moves the whole trace so its root starts at that time (LangSmith
    only ingests runs that start within 24 hours of now); the original start time
    is kept in the root run's metadata as langfuse_start_time.
    """
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
    # LangSmith groups runs into threads by a session_id / thread_id metadata key,
    # which has to be on every run of the trace.
    session_id = _get_attr(source_trace, ['session_id', 'sessionId'])
    root["metadata"] = dict(root["metadata"])
    if session_id:
        root["metadata"]["session_id"] = str(session_id)
    user_id = _get_attr(source_trace, ['user_id', 'userId'])
    if user_id:
        root["metadata"]["user_id"] = str(user_id)
    root["error"] = _error_of(source_trace)
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
        if run_type == "llm":
            inputs = _to_messages(inputs, default_role="user")
            outputs = _to_messages(outputs, default_role="assistant")
        parent_obs_id = _get_attr(obs, ['parent_observation_id','parentObservationId'])
        # Unknown parents (e.g. the root span folded into the trace) attach to the root run.
        parent_run_id = obs_id_to_run_id.get(str(parent_obs_id)) or root["id"] if parent_obs_id else root["id"]
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
            "error": _error_of(obs),
        }
        run["metadata"] = dict(run["metadata"])
        if session_id:
            run["metadata"]["session_id"] = str(session_id)
        if run_type == "llm":
            model = _get_attr(obs, ['model'])
            if model:
                # ls_model_name is what LangSmith uses to price the run
                run["metadata"]["ls_model_name"] = str(model)
                run["metadata"]["ls_provider"] = _provider_from_model(str(model))
            usage_md = _usage_metadata(obs)
            if usage_md:
                if run["outputs"] is None:
                    run["outputs"] = {}
                if isinstance(run["outputs"], dict):
                    run["outputs"]["usage_metadata"] = usage_md
        runs.append(run)
    # Finish: end times + dotted order
    lf_trace_id = _get_attr(source_trace, ['id'])
    if lf_trace_id:
        root["metadata"]["langfuse_trace_id"] = str(lf_trace_id)
    if root.get("start_time"):
        root["metadata"]["langfuse_start_time"] = root["start_time"]
    _ensure_end_times(runs)
    if rebase_to is not None:
        _rebase_times(runs, rebase_to)
    _assign_dotted_order(runs, root["id"])
    return runs


# OpenTelemetry bookkeeping that Langfuse v4 copies into observation metadata.
_OTEL_METADATA_PREFIXES = ("resourceAttributes.", "attributes.", "scope.")


def _parse_io(val):
    """v2 observations return input/output as raw JSON strings."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except ValueError:
            return val
    return val


def _clean_metadata(md) -> dict:
    md = _parse_io(md)
    if not isinstance(md, dict):
        return {} if md in (None, "") else {"metadata": md}
    return {k: v for k, v in md.items() if not str(k).startswith(_OTEL_METADATA_PREFIXES)}


def trace_from_observations(observations: list[dict]) -> dict:
    """Rebuild a v3-style trace (what map_langfuse_to_langsmith reads) from the
    v2 observations of one trace.

    In Langfuse v4 a trace has no input/output of its own: they come from its root
    observation. When there is one root span, it becomes the LangSmith root run
    (its name, input, output, error) and its children hang off that run, so the
    span is not nested twice.
    """
    obs = []
    for o in observations:
        o = dict(o)
        o["input"] = _parse_io(o.get("input"))
        o["output"] = _parse_io(o.get("output"))
        o["metadata"] = _clean_metadata(o.get("metadata"))
        obs.append(o)
    obs.sort(key=lambda o: o.get("startTime") or "")
    roots = [o for o in obs if not o.get("parentObservationId")]
    ctx = roots[0] if roots else obs[0]

    def first(key):
        return next((o.get(key) for o in obs if o.get(key)), None)

    trace = {
        "id": ctx.get("traceId"),
        "name": ctx.get("traceName") or ctx.get("name"),
        "sessionId": first("sessionId"),
        "userId": first("userId"),
        "tags": ctx.get("tags") or [],
        "timestamp": ctx.get("startTime"),
        "metadata": dict(ctx.get("metadata") or {}),
        "input": None,
        "output": None,
    }
    if ctx.get("release"):
        trace["metadata"]["release"] = ctx["release"]

    if len(roots) == 1 and str(roots[0].get("type")).upper() not in ("GENERATION", "EMBEDDING"):
        root = roots[0]
        trace.update(
            name=root.get("name") or trace["name"],
            input=root["input"],
            output=root["output"],
            endTime=root.get("endTime"),
            level=root.get("level"),
            statusMessage=root.get("statusMessage"),
        )
        trace["observations"] = [o for o in obs if o is not root]
    else:
        if len(roots) == 1:
            trace.update(input=roots[0]["input"], output=roots[0]["output"])
        trace["observations"] = obs
    return trace


def _legacy_recent_trace_ids(max_traces: int | None) -> list[str]:
    """Trace IDs from the v3 traces API, for self-hosted servers without v2 observations."""
    ids: list[str] = []
    page = 1
    while max_traces is None or len(ids) < max_traces:
        listing = lf_get("/api/public/traces", page=page, limit=50)
        objs = (
            listing.get("objects") if isinstance(listing, dict) and "objects" in listing
            else listing.get("data") if isinstance(listing, dict) else listing
        ) or []
        if not objs:
            break
        ids.extend(str(o["id"]) for o in objs if o.get("id"))
        page += 1
    return ids[:max_traces] if max_traces is not None else ids


def fetch_and_transform_traces(workspace_id: str, sleep_between_gets=0.7, max_retries=4):
    """Replay the most recent NUM_TRACES_TO_REPLAY Langfuse traces into LangSmith.

    Reads from the v2 observations API (Langfuse v4 and Langfuse Cloud). Older
    self-hosted servers without it are read through the v3 traces API.
    """
    try:
        max_traces = int(NUM_TRACES_TO_REPLAY)
    except Exception:
        max_traces = None

    if max_traces == 0:
        print("        ! NUM_TRACES_TO_REPLAY is 0 (or unset), so no traces will be replayed. Set it in .env.")
        return

    try:
        if lf_supports_observations_v2():
            print("        • reading traces from the Langfuse v2 observations API")
            trace_ids = lf_recent_trace_ids(max_traces)
            fetch = lambda tid: trace_from_observations(lf_get_trace_observations(tid))
        else:
            print("        • no v2 observations API on this Langfuse server, using the v3 traces API")
            trace_ids = _legacy_recent_trace_ids(max_traces)
            fetch = lambda tid: lf_get(f"/api/public/traces/{tid}")
    except Exception as e:
        print(f"        x couldn't list Langfuse traces, no traces were migrated: {e}")
        return

    total_processed = 0
    total_failed_fetch = 0
    total_failed_transform = 0
    sources: list[dict] = []
    for tid in trace_ids:
        source = None
        for attempt in range(max_retries):
            time.sleep(sleep_between_gets * (2 ** attempt))
            try:
                source = fetch(tid)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"        ! couldn't fetch trace {tid}: {e}")
        if not isinstance(source, dict):
            total_failed_fetch += 1
            continue
        sources.append(source)

    # LangSmith rejects runs that start more than 24 hours from now. If any trace
    # is older than that, lay the whole batch out on a compressed timeline ending
    # now (one second apart, oldest first) so order and session turn order hold.
    def _start(s):
        return _parse_ts(_get_attr(s, ['timestamp', 'start_time', 'startTime'])) or dt.datetime.now(dt.timezone.utc)

    sources.sort(key=_start)
    now = dt.datetime.now(dt.timezone.utc)
    rebase = bool(sources) and now - _start(sources[0]) > dt.timedelta(hours=23)
    if rebase:
        print(
            "        • traces older than 24 hours: LangSmith only accepts runs that start within "
            "24 hours of now, so start times are moved to the last few minutes (order and "
            "durations kept; the original time is in metadata.langfuse_start_time)"
        )

    accumulated_runs: list[dict] = []
    for i, source in enumerate(sources):
        tid = _get_attr(source, ['id'])
        new_start = now - dt.timedelta(seconds=len(sources) - i + 60) if rebase else None
        try:
            runs_batch = map_langfuse_to_langsmith(source, rebase_to=new_start)
        except Exception as e:
            print(f"        ! couldn't convert trace {tid}: {e}")
            total_failed_transform += 1
            continue
        accumulated_runs.extend(runs_batch)
        total_processed += 1

    if accumulated_runs:
        try:
            ls_replay_runs_sdk(workspace_id, accumulated_runs)
        except Exception as e:
            print(f"        x uploading runs to LangSmith failed: {e}")
            total_processed = 0

    print(f"        • Processed traces: {total_processed} of {len(trace_ids)} found")
    print(f"        • Failed fetching details (after retries): {total_failed_fetch}")
    print(f"        • Failed transforming data: {total_failed_transform}")


def migrate_traces(workspace_id: str, project_id: str):
    print(f"    - migrating recent traces…")
    fetch_and_transform_traces(workspace_id)
