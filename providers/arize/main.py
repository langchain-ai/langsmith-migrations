from utils.langsmith import ls_get_or_create_workspace
from config import LS_WORKSPACE_ID

from providers.arize.data.datasets import migrate_datasets
from providers.arize.data.traces import migrate_traces


def migrate_arize(projects: list[dict]):
    """Migrate data from Arize to LangSmith."""
    
    for proj in projects:
        project_name = proj.get("name")
        print(f"\n-  Project: {project_name}")
        
        if LS_WORKSPACE_ID:
            ws_id = LS_WORKSPACE_ID
            print(f"    + using workspace: {ws_id}")
        else:
            ws = ls_get_or_create_workspace(project_name)
            ws_id = ws["id"]
            print(f"    + workspace id: {ws_id}")

        migrate_datasets(ws_id)
        migrate_traces(ws_id, project_name)

    print("\n+  Migration complete.")
