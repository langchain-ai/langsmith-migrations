import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from urllib.parse import quote

from config import LF_BASE, LF_HEADERS, LF_PUBLIC_KEY, LF_SECRET_KEY


def lf_get(path: str, **params):
    """GET helper for Langfuse Public/Admin API with Basic auth."""
    resp = requests.get(
        f"{LF_BASE}{path}",
        headers=LF_HEADERS,
        params=params,
        auth=HTTPBasicAuth(LF_PUBLIC_KEY or "", LF_SECRET_KEY or ""),
    )
    resp.raise_for_status()
    return resp.json()


def lf_get_projects() -> list[dict]:
    """List Langfuse projects.

    Tries Public API path first, then falls back to Admin API path.
    """
    try:
        # Public API style
        data = lf_get("/api/public/projects")
        # Normalize to list of {id, name}
        if isinstance(data, dict):
            items = data.get("data") or data.get("projects") or data.get("objects")
        else:
            items = data
        if items:
            return items
    except Exception as e:
        raise e



def lf_get_project_prompts(project_id: str) -> list[dict]:
    """
    Return prompts for a Langfuse project.
    """
    names: list[str] = []
    page = 1
    while True:
        listing = lf_get("/api/public/v2/prompts", project_id=project_id, page=page)
        objs = (
            listing.get("objects")
            if isinstance(listing, dict) and "objects" in listing
            else listing.get("data") if isinstance(listing, dict) else listing
        ) or []
        if not objs:
            break
        for p in objs:
            name = p.get("name") or p.get("slug") or p.get("id")
            if name:
                names.append(str(name))
        page += 1

    detailed: list[dict] = []
    for name in names:
        try:
            p_detail = lf_get(f"/api/public/v2/prompts/{quote(name, safe='')}", project_id=project_id)
            detailed.append(p_detail)
        except Exception:
            detailed.append({"name": name})
    return detailed


def lf_get_prompt_history(project_id: str) -> list[dict]:
    """Return every version of every prompt, for migrating version history.

    Result: [{"name": str, "versions": [prompt detail dicts], "failed_versions": [int]}].
    The listing reports each prompt's version numbers; each one is then fetched with
    ?version=N so it carries its own content, config, labels and commit message.
    If the listing does not report versions, only the default (production-labelled)
    version is fetched, as lf_get_project_prompts does.
    """
    metas: list[dict] = []
    page = 1
    while True:
        listing = lf_get("/api/public/v2/prompts", project_id=project_id, page=page)
        objs = (
            listing.get("objects")
            if isinstance(listing, dict) and "objects" in listing
            else listing.get("data") if isinstance(listing, dict) else listing
        ) or []
        if not objs:
            break
        metas.extend(objs)
        page += 1

    history: list[dict] = []
    for meta in metas:
        name = meta.get("name") or meta.get("slug") or meta.get("id")
        if not name:
            continue
        name = str(name)
        path = f"/api/public/v2/prompts/{quote(name, safe='')}"
        numbers = sorted({int(v) for v in (meta.get("versions") or [])})
        versions: list[dict] = []
        failed: list = []
        if numbers:
            for n in numbers:
                try:
                    versions.append(lf_get(path, project_id=project_id, version=n))
                except Exception as e:
                    print(f"      ! couldn't fetch '{name}' version {n}: {e}")
                    failed.append(n)
        else:
            try:
                versions.append(lf_get(path, project_id=project_id))
            except Exception as e:
                print(f"      ! couldn't fetch '{name}': {e}")
                failed.append("default")
        history.append({"name": name, "versions": versions, "failed_versions": failed})
    return history


# Field groups of GET /api/public/v2/observations needed to rebuild a trace.
_OBSERVATION_FIELDS = "core,basic,time,io,metadata,model,usage,prompt,trace_context"


def lf_iter_observations(**params):
    """Yield observations from GET /api/public/v2/observations, following the cursor.

    Langfuse v4 reads traces through this endpoint; the v3 /api/public/traces
    endpoints are removed on Langfuse Cloud (410 Gone since 2026-11-16).
    Results come newest first.
    """
    cursor = None
    while True:
        query = dict(params)
        if cursor:
            query["cursor"] = cursor
        page = lf_get("/api/public/v2/observations", **query)
        rows = page.get("data") or []
        yield from rows
        cursor = (page.get("meta") or {}).get("cursor")
        if not cursor or not rows:
            return


def lf_supports_observations_v2() -> bool:
    """False on Langfuse servers that predate the v2 observations API (404)."""
    try:
        lf_get("/api/public/v2/observations", limit=1)
        return True
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return False
        raise


def lf_recent_trace_ids(max_traces: int | None) -> list[str]:
    """IDs of the most recent traces, newest first, found through their root observations."""
    ids: list[str] = []
    seen: set[str] = set()
    for obs in lf_iter_observations(isRootObservation="true", fields="core", limit=1000):
        tid = obs.get("traceId")
        if tid and tid not in seen:
            seen.add(tid)
            ids.append(tid)
            if max_traces is not None and len(ids) >= max_traces:
                break
    return ids


def lf_get_trace_observations(trace_id: str) -> list[dict]:
    """Every observation of one trace, with input/output, usage, model and trace context."""
    return list(
        lf_iter_observations(
            traceId=trace_id, fields=_OBSERVATION_FIELDS, limit=1000
        )
    )


def lf_get_project_datasets(
    project_id: str
) -> dict[str, list[dict]]:
    """Fetch datasets and their records from Langfuse for a project.

    Returns a mapping of dataset name → list of record dicts.
    """
    out: dict[str, list[dict]] = {}
    # List datasets (v2) with page/limit pagination
    datasets: list[dict] = []
    page = 1
    while True:
        dresp = lf_get("/api/public/v2/datasets", project_id=project_id, page=page, limit=100)
        batch = (
            dresp.get("objects")
            if isinstance(dresp, dict) and "objects" in dresp
            else dresp.get("data") if isinstance(dresp, dict) else dresp
        ) or []
        if not batch:
            break
        datasets.extend(batch)
        page += 1

    for ds in datasets:
        ds_id = ds.get("id")
        ds_name = ds.get("name") or f"dataset-{ds_id}"
        items: list[dict] = []
        try:
            # List dataset items (v2) with page/limit; then enrich each with detail by id
            page_items = 1
            while True:
                listing = lf_get(
                    "/api/public/dataset-items",
                    datasetId=ds_id,
                    page=page_items,
                    limit=100,
                )
                batch = (
                    listing.get("objects")
                    if isinstance(listing, dict) and "objects" in listing
                    else listing.get("data") if isinstance(listing, dict) else listing
                ) or []
                if not batch:
                    break

                for it in batch:
                    item_id = it.get("id") or it.get("item_id") or it.get("uuid")
                    if not item_id:
                        items.append(it)
                        continue
                    try:
                        detail = lf_get(f"/api/public/dataset-items/{item_id}")
                        items.append(detail or it)
                    except Exception:
                        items.append(it)

                page_items += 1

            print(f"      > fetched {len(items)} items from '{ds_name}'")
        except Exception as e:
            print(f"      ! couldn't fetch '{ds_name}': {e}")
            items = []
        out[ds_name] = items
    return out
