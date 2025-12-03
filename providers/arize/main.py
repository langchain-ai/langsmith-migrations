from utils.langsmith import ls_get_or_create_workspace
from config import LS_WORKSPACE_ID

from providers.arize.data.datasets import migrate_datasets
from providers.arize.data.traces import migrate_traces


def migrate_arize(space_name: str = "arize"):
    """Migrate data from Arize to LangSmith."""
    
    if LS_WORKSPACE_ID:
        ws_id = LS_WORKSPACE_ID
        print(f"\n-  Using workspace: {ws_id}")
    else:
        print(f"\n-  Arize Space → LangSmith Workspace: {space_name}")
        ws = ls_get_or_create_workspace(space_name)
        ws_id = ws["id"]
    
    print(f"    + workspace id: {ws_id}")

    migrate_datasets(ws_id)
    migrate_traces(ws_id)

    print("\n+  Migration complete.")
