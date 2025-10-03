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
