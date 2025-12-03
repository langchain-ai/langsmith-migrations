# Migrating to LangSmith

### Setup

1. Create a `.env` file in the project root following the .env.example file

2. Use a Python virtual environment and install dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Migrating Prompts, Datasets, and Traces
The migration orchestrator prompts you to select which observability framework you want to migrate from.
```bash
source .venv/bin/activate
python -u migrate.py
```

1. Select the desired observability provider to migrate from
2. Confirm migration settings with `yes` or `no` to migrate prompts, datasets, and traces.
    - Note: Trace replay is for illustrative purposes on how to convert formats. Migrating your full trace data to LangSmith should be done using bulk export functionality.


## Runbooks (Notebooks)
Each provider has a runbook notebook to illustrate how to migrate your existing code to utilize LangSmith.

Each observability provider in this repo has a runbook under its corresponding directory.
- Langfuse runbook: `providers/langfuse/runbook.ipynb`
- Arize runbook: `providers/arize/runbook.ipynb` (coming soon)

Usage:
1. Open the runbook in your IDE or using ```jupyter notebook``` in the root directory.
2. Review sections for prompts, datasets, and traces.
3. Execute cells to inspect transformations and payloads.
