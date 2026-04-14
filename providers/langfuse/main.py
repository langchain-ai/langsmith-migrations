from utils.langsmith import ls_get_or_create_workspace
from config import LS_WORKSPACE_ID

from providers.langfuse.data.prompts import migrate_prompts
from providers.langfuse.data.datasets import migrate_datasets
from providers.langfuse.data.traces import migrate_traces


def migrate_langfuse(projects: list[dict]):
    # Migrate selected projects
    for proj in projects:
        pname = proj.get("name") or proj.get("display_name") or proj.get("slug") or str(proj.get("id"))
        pid = proj.get("id") or proj.get("project_id") or proj.get("uuid") or ""
        print(f"\n-  Project: {pname}")

        if LS_WORKSPACE_ID:
            ws_id = LS_WORKSPACE_ID
            print(f"    + using workspace: {ws_id}")
        else:
            ws = ls_get_or_create_workspace(pname)
            ws_id = ws["id"]
            print(f"    + workspace id: {ws_id}")

        migrate_prompts(ws_id, pid)
        migrate_datasets(ws_id, pid)
        migrate_traces(ws_id, pid)
                
    print("\n+  Migration complete.")