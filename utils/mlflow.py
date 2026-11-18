"""MLflow connection and bounded discovery helpers shared by provider resources."""
from stack_common import is_databricks_uri, databricks_registry_uri

def bounded(items, limit, resource):
    rows = list(items)
    if len(rows) >= limit or getattr(items, 'token', None):
        raise ValueError(f'{resource} reached discovery bound; narrow scope before export.')
    return rows


def connect(tracking_uri):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    if is_databricks_uri(tracking_uri):
        mlflow.set_registry_uri(databricks_registry_uri(tracking_uri))
    return mlflow, mlflow.MlflowClient()
