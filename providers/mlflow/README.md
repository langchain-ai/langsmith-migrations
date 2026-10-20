# MLflow → LangSmith

Start with [runbook.ipynb](runbook.ipynb), matching the Phoenix/Langfuse structure:
**Migrating Resources**, then **Updating Code**. The latter generates new traces
and experiments. It does not recreate historical experiments.

## Execution

Run from the repository root with Python 3.12 and dependencies from
`providers/mlflow/requirements.txt`. Use the root `migrate.py` provider selector,
or `python migrate.py --provider mlflow` with arguments:

```sh
python migrate.py --provider mlflow --list-experiments --tracking-uri databricks://PROFILE
python migrate.py --provider mlflow --experiment ID --tracking-uri databricks://PROFILE --output migration-export
python migrate.py --provider mlflow --bundle migration-export/bundle.json --output migration-export --apply
```

Locally, authenticate using `databricks auth login --host HOST --profile PROFILE`.
On Databricks, use tracking URI `databricks` with notebook workspace authentication.
The self-contained [companion notebook](03_databricks_export_migrate.ipynb) supports
both environments. Source experiment discovery does not require a setup manifest.

Destination configuration: `LANGSMITH_API_KEY`, explicit `LANGSMITH_WORKSPACE_ID`,
and optional `LANGSMITH_ENDPOINT`. Load an existing `.env` into the process before
calling the provider if desired; never embed credentials in notebook outputs.
Interactive defaults: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`,
`MLFLOW_MIGRATION_OUTPUT`, and `MLFLOW_MIGRATION_PREFIX` in root `config.py`.

UC trace reads need a SQL warehouse. The exporter uses the experiment monitoring
warehouse tag when available; otherwise set `MLFLOW_TRACING_SQL_WAREHOUSE_ID`.
The MLflow SDK can start the warehouse to serve queries.

## Supported resources and limits

| Resource | Behavior |
|---|---|
| Dataset examples | Preserve inputs, reference expectations and provenance; keep datasets separate. |
| Text prompt versions | Private snapshots, with simple variable conversion; complex templates are reported as skipped. |
| Traces | Preserve hierarchy, payloads, errors and original source attributes. |
| Feedback | Map supported values and retain source provenance; errors/expectations remain archived. |
| Experiments / scorers | Archive run metadata and scorer configuration. No general historical experiment reconstruction or live scorer binding. |
| Agent visibility | Attach imported datasets and prompts through the reserved Agent resource association. Application tags alone are insufficient. |
| Ongoing use | Runbook demonstrates fresh tracing, datasets, evaluations and prompts. |

UC prompt discovery follows recorded run/trace links; unlinked prompts are outside
its scope. Arbitrary artifacts, historical dataset versions, aliases, serving,
permissions, annotation queues, online schedules, Insights and Engine are not
recreated. Imported token/cost attributes are not normalized billing estimates.

The tested workspace rejected trace starts outside ±24 hours. Opt into replay:

```sh
python migrate.py --provider mlflow --bundle migration-export/bundle.json --output replay-export --prefix mlflow-replay --replay-timestamps --apply
```

Replay retains source dates in metadata and intra-trace durations, but changes
displayed dates and gaps between traces. Preserve `timestamp-policy.json` and
checkpoints. Use a fresh output directory and prefix when changing replay policy.
Do not rerun source export into an already-applied snapshot.

## Validation

Exports save coverage reports and block import on discovery errors. Import performs
read-back validation. Reruns use stable IDs and checkpoints. Resource associations
are also read back. Local tests: `python -m unittest discover -s tests -q`.
The previously executed live validation covered 87 traces / 373 spans, 203 feedback
records, 12 examples and two prompts, plus a duplicate-free rerun. The separate
new-activity examples exercised a tool trace, three examples/evaluations and prompt
rendering. These checks validate API records and associations, not browser rendering.

## Separate development fixtures

Synthetic source setup helpers live in `tests/seed_mlflow.py` and
`tests/extend_mlflow.py` for regression coverage. Migration never invokes them.
The optional `--manifest` adapter supports existing synthetic inventories;
normal migrations use `--experiment`. Source setup notebooks and workspace-specific
exports are not included in this contribution.
