"""Discover MLflow datasets and migrate their reference examples."""
from stack_common import digest, plain, is_databricks_uri
from utils.mlflow import bounded


def export_datasets(mlflow, experiment_id, tracking_uri, names=(), limit=10000):
    found = bounded(mlflow.genai.datasets.search_datasets(
        experiment_ids=[str(experiment_id)], max_results=limit,
        **({} if is_databricks_uri(tracking_uri) else {'filter_string': 'created_time >= 0'})),
        limit, 'datasets')
    by_name = {ds.name: ds for ds in found}
    for name in names:
        if name not in by_name:
            by_name[name] = mlflow.genai.datasets.get_dataset(name=name)
    datasets = []
    for name, ds in sorted(by_name.items()):
        # Search returns tracking-store entities; UC rows must be loaded by name.
        if is_databricks_uri(tracking_uri):
            ds = mlflow.genai.datasets.get_dataset(name=name)
        dataset = {'name': name, 'native': {'name': name, 'id': ds.dataset_id}, 'records': []}
        for idx, row in enumerate(plain(ds.to_df().to_dict(orient='records'))):
            inputs, refs = row.get('inputs'), row.get('expectations')
            if not isinstance(inputs, dict) or not isinstance(refs, dict):
                raise ValueError('Dataset records require dictionary inputs and expectations; predictions are not references.')
            dataset['records'].append({
                'id': str(row.get('dataset_record_id') or row.get('id') or digest([idx, row])),
                'inputs': inputs, 'expectations': refs,
                'metadata': {'source_dataset': name}, 'native_record': row})
        datasets.append(dataset)
    return datasets


def migrate_datasets(plan, destination, perform):
    dataset_id = (perform('dataset', lambda: destination.dataset(plan))
                  if plan.get('create_dataset', True) else None)
    for item in plan['examples']:
        perform('example:' + item['id'], lambda i=item: destination.example(i, dataset_id))
    return dataset_id
