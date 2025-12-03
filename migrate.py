from providers.langfuse.main import migrate_langfuse
from providers.arize.main import migrate_arize
from config import INCLUDE_MODEL_IN_PROMPTS, NUM_TRACES_TO_REPLAY
from utils.langfuse import lf_get_projects

AVAILABLE_PROVIDERS = [
    "langfuse",
    "arize",
]

## ------------------------------------------------------------
## Helpers
## ------------------------------------------------------------
def display_config(name: str, resource_name: str):
    print("\n")
    print("╭─────────────────────────────────────────────────────────────╮")
    print("│                    MIGRATION SETTINGS                       │")
    print("╰─────────────────────────────────────────────────────────────╯")
    print(f"-  Migrate {name} {resource_name}s")
    print(
        f"-  Include models in prompts: {'+ Yes' if INCLUDE_MODEL_IN_PROMPTS else '- No'}"
        f"\n-  Number of traces to replay: {NUM_TRACES_TO_REPLAY}"
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
    elif provider == "arize":
        display_config("Arize", "space")
        should_migrate = capture_user_selection("Arize", "datasets")
        if not should_migrate:
            return
        migrate_arize()   
    
    
def prompt_for_provider() -> str:
    print("\nSelect a provider to migrate from:")
    for idx, p in enumerate(AVAILABLE_PROVIDERS, start=1):
        print(f"  {idx}. {p}")
    while True:
        raw = input("\nEnter number or name (q to quit): ").strip().lower()
        if raw == "q":
            return None
        # number selection
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(AVAILABLE_PROVIDERS):
                return AVAILABLE_PROVIDERS[i - 1]
        # name selection
        if raw in AVAILABLE_PROVIDERS:
            return raw
        print("Invalid selection. Please choose a valid number or name from the list.")


if __name__ == "__main__":
    provider = prompt_for_provider()
    if not provider:
        print("\nMigration cancelled. Exiting.")
    else:   
        migrate(provider)