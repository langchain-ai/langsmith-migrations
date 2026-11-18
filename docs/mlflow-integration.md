# MLflow contribution workflow

1. Prepare the contribution on `mlflow-migration`. The provider follows the
   existing `main.py`, `data/`, and `runbook.ipynb` layout. Root discovery,
   transformation, checkpoint, and validation helpers support both execution paths.
2. Run `python -m unittest discover -s tests -q` from the repository root.
   This includes offline notebook execution and CLI preview. Live checks require
   Databricks source access and a dedicated LangSmith destination workspace.
3. Review the diff. Keep source exports, checkpoints, credentials, local tools,
   and workspace-specific reports out of the contribution. Synthetic setup helpers
   under `tests/` are fixtures, not prerequisites for migration.
4. Commit and push the reviewed changes to `mlflow-migration`, then open a pull
   request against `main`. Describe supported datasets, prompts, traces, feedback,
   timestamp replay, and validation results. Link the provider runbook.
5. Address maintainer feedback and required checks, then merge through the
   repository review process. After merge, users can choose MLflow through the
   root dispatcher or follow the provider runbook.

The runbook separates importing existing resources from generating new LangSmith
activity. Historical evaluation experiments, live scorer rules, review workflows,
and serving deployments are not reconstructed. Local execution uses a Databricks
CLI profile; execution on Databricks uses workspace authentication.
