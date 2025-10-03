from providers.langfuse.main import migrate_langfuse
from config import INCLUDE_MODEL_IN_PROMPTS, NUM_TRACES_TO_REPLAY
from utils.langfuse import lf_get_projects

## ------------------------------------------------------------
## Helpers
## ------------------------------------------------------------
def display_config(name: str, resource_name: str):
    print("╭─────────────────────────────────────────────────────────────╮")
    print("│                    MIGRATION SETTINGS                       │")
    print("╰─────────────────────────────────────────────────────────────╯")
    print(f"-  Migrate {name} {resource_name}s")
    print(
        f"-  Include models in prompts: {'+ Yes' if INCLUDE_MODEL_IN_PROMPTS else '- No'}"
        f"-  Number of traces to replay: {NUM_TRACES_TO_REPLAY}"
    )
    print("─" * 63)


def capture_user_selection(name: str,resource_name: str):
    # Get user selection
    while True:
        choice = input(
            f"Migrate the {resource_name} associated with your {name} API key? (y/n): "
        ).strip().lower()
            
        if choice == 'no' or choice == 'n':
            print("Migration cancelled.")
            return False
        elif choice == 'yes' or choice == 'y':
            break
        else:
            print("Invalid input. Please enter 'yes or 'no'")
            continue
    return True


## ------------------------------------------------------------
## Main Migration Function
## ------------------------------------------------------------
def migrate(provider: str):
    if provider == "langfuse":
        display_config("Langfuse", "project")
        migrate = capture_user_selection("Langfuse", "project")
        projects = lf_get_projects()
        if not migrate:
            return
        if not projects:
            print("No project found for the configured API keys.")
            return
        migrate_langfuse(projects)   
    else:
        print("Invalid provider.\nSupported providers are: 'langfuse'")
    

if __name__ == "__main__":
    migrate("langfuse")