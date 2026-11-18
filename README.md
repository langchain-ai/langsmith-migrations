# Migrating to LangSmith

Migrate prompts, datasets, and traces from Langfuse, Arize Phoenix, or MLflow,
then update your application to use LangSmith.

## Setup

1. Create a `.env` file in the project root using `.env.example` as a reference.
   Configure credentials for your source provider and LangSmith destination.
2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies for your source provider:

   | Source provider | Install command |
   | --- | --- |
   | Langfuse | `pip install -r requirements.txt` |
   | Arize Phoenix | `pip install -r requirements.txt` |
   | MLflow | `pip install -r providers/mlflow/requirements.txt` |

   For MLflow, use Python 3.12 and follow the [provider setup guide](providers/mlflow/README.md)
   for local Databricks CLI authentication or execution directly on Databricks.

## Migrating Prompts, Datasets, and Traces

Run the migration orchestrator from the repository root:

```bash
source .venv/bin/activate
python -u migrate.py
```

1. Select your source provider: Langfuse, Arize Phoenix, or MLflow.
2. Review the migration settings and follow the provider's confirmation prompts.

Supported resources, trace replay behavior, and migration limits vary by provider.
Review its runbook before importing data. The Langfuse and Arize Phoenix trace
replay examples illustrate format conversion; use bulk export functionality when
planning a full trace migration.

## Runbooks (Notebooks)

Each provider has a runbook covering resource migration and updating application
code to use LangSmith.

| Source provider | Runbook | Additional guidance |
| --- | --- | --- |
| Langfuse | [Runbook](providers/langfuse/runbook.ipynb) | — |
| Arize Phoenix | [Runbook](providers/phoenix/runbook.ipynb) | — |
| MLflow | [Runbook](providers/mlflow/runbook.ipynb) | [Setup and limitations](providers/mlflow/README.md) · [Standalone Databricks/local migration notebook](providers/mlflow/03_databricks_export_migrate.ipynb) |

1. Open the runbook in your IDE or Jupyter from the repository root.
2. Review its configuration, supported resources, and limitations.
3. Execute the migration examples, then follow the application instrumentation
   and new-activity examples.
