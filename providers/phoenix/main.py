from utils.langsmith import ls_get_or_create_workspace
from config import LS_WORKSPACE_ID

from providers.phoenix.data.prompts import migrate_prompts
from providers.phoenix.data.datasets import migrate_datasets
from providers.phoenix.data.traces import migrate_traces


def migrate_phoenix(projects: list[dict]):
    """Migrate data from Phoenix to LangSmith."""
    
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

        migrate_prompts(ws_id)
        migrate_datasets(ws_id)
        migrate_traces(ws_id, project_name)

    print("\n+  Migration complete.")

