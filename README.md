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

## Langfuse notes

What the Langfuse migration does and does not carry over:

- **Prompts**: every version of every prompt is pushed to LangSmith oldest first, so it
  keeps the same commit history. Langfuse labels (except the automatic `latest`),
  including `production` and `staging`, are applied as commit tags; a tag LangSmith
  refuses is reported with the reason. A version identical to the one before it (for
  example a re-publish) adds no commit; its labels go on the previous commit.
  Prompts use the mustache format. Variables become `{{{variable}}}` so values are
  inserted as given, as in Langfuse (mustache's `{{variable}}` would HTML-escape them).
  Chat placeholders become optional message placeholders. Prompt names are lowercased
  and any character other than letters, digits, `-` and `_` becomes `-`; the script
  prints each Langfuse name next to its LangSmith name, and skips a prompt whose
  LangSmith name is already taken by another prompt in the same run. A prompt that
  already exists in LangSmith is skipped, so re-runs do not add duplicate commits.
  If a run stops partway through a prompt, delete that prompt in LangSmith and re-run.
- **Datasets**: dataset items are copied as examples. Experiments, dataset runs, scores
  and dataset versions are not migrated.
- **Traces**: only the most recent `NUM_TRACES_TO_REPLAY` traces are replayed, and the
  default is 0. Traces are read from the v2 observations API (Langfuse v4 and Langfuse
  Cloud, where the v3 `/api/public/traces` endpoints return 410 since 2026-11-16); older
  self-hosted servers without it are read through the v3 traces API. Models, token usage
  (cached prompt tokens included), cost, errors, tags, user ID and session ID are mapped,
  so sessions become threads. LangSmith only accepts runs that start within 24 hours of
  now, so when the batch holds older traces their start times are moved to the last few
  minutes, keeping their order and durations; the original start time and Langfuse trace
  ID are in each root run's metadata (`langfuse_start_time`, `langfuse_trace_id`). Runs
  get new IDs. Scores and experiments (dataset runs) are not migrated. For a full history
  use Langfuse's blob storage export, or keep Langfuse read-only after cutover.
- Not covered: evaluators, annotation queues, users and roles, API keys.

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
