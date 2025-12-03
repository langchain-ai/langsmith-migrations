import pandas as pd
from datetime import datetime, timedelta, timezone
from arize.experimental.datasets import ArizeDatasetsClient
from arize.exporter import ArizeExportClient
from arize.utils.types import Environments

from config import ARIZE_API_KEY, ARIZE_SPACE_ID, ARIZE_PROJECT_NAME


def get_arize_client() -> ArizeDatasetsClient:
    """Get an authenticated Arize Datasets client."""
    return ArizeDatasetsClient(
        api_key=ARIZE_API_KEY,
    )


def arize_get_datasets() -> pd.DataFrame:
    """List all datasets in the Arize space.

    Returns a DataFrame with dataset metadata (id, name, etc.).
    """
    client = get_arize_client()
    datasets_df = client.list_datasets(space_id=ARIZE_SPACE_ID)
    return datasets_df


def arize_get_dataset_examples(dataset_id: str = None, dataset_name: str = None) -> pd.DataFrame:
    """Fetch all examples from an Arize dataset.

    Args:
        dataset_id: The dataset ID. Required if dataset_name is not provided.
        dataset_name: The dataset name. Required if dataset_id is not provided.

    Returns a DataFrame with the dataset examples.
    """
    client = get_arize_client()
    examples_df = client.get_dataset(
        space_id=ARIZE_SPACE_ID,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
    )
    return examples_df


def get_arize_export_client() -> ArizeExportClient:
    """Get an authenticated Arize Export client for traces."""
    return ArizeExportClient()


def arize_export_traces(days_back: int = 7) -> pd.DataFrame:
    """Export traces from Arize.

    Args:
        days_back: Number of days back to export traces from (default: 7).

    Returns a DataFrame with trace/span data.
    """
    client = get_arize_export_client()
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    
    traces_df = client.export_model_to_df(
        space_id=ARIZE_SPACE_ID,
        model_id=ARIZE_PROJECT_NAME,
        environment=Environments.TRACING,
        start_time=start_time,
        end_time=end_time,
    )
    return traces_df
