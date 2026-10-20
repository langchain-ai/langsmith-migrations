"""Read-only, experiment-scoped discovery, independent of the setup inventory."""
from copy import deepcopy
import os

from stack_common import SCHEMA_VERSION, digest, plain, save_json, is_databricks_uri, databricks_registry_uri


from utils.mlflow import bounded, connect
from providers.mlflow.data.datasets import export_datasets
from providers.mlflow.data.prompts import export_prompts
from providers.mlflow.data.traces import export_traces


def list_experiments(tracking_uri='databricks'):
    _, client = connect(tracking_uri)
    rows, token = [], None
    while True:
        page = client.search_experiments(max_results=100, page_token=token)
        rows.extend({'id': e.experiment_id, 'name': e.name} for e in page)
        token = getattr(page, 'token', None)
        if not token:
            return rows


def export_experiment(experiment_id, output_path, tracking_uri='databricks',
                      dataset_names=(), prompt_names=(), limit=10000, source_key=None):
    """Export a portable collection: one bundle per dataset and one for traces/prompts.

    Explicit dataset/prompt names augment discovery; they never suppress discovery errors.
    Unsupported APIs are coverage gaps, not silently reported as empty collections.
    """
    if limit < 2:
        raise ValueError('limit must be at least 2')
    mlflow, client = connect(tracking_uri)
    experiment = client.get_experiment(str(experiment_id))
    # Include source host in identity; profile aliases alone are not stable identities.
    if source_key is None:
        if is_databricks_uri(tracking_uri):
            from databricks.sdk import WorkspaceClient
            from mlflow.utils.uri import get_db_info_from_uri
            profile, _ = get_db_info_from_uri(tracking_uri)
            ws = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
            source_key = ws.config.host
        else:
            source_key = tracking_uri
    warehouse = os.environ.get('MLFLOW_TRACING_SQL_WAREHOUSE_ID') or (experiment.tags or {}).get('mlflow.monitoring.sqlWarehouseId')
    if warehouse:
        os.environ['MLFLOW_TRACING_SQL_WAREHOUSE_ID'] = warehouse
    source = {'seed_id': digest([source_key, str(experiment_id)])[:24],
              'experiment_id': str(experiment_id), 'experiment_name': experiment.name,
              'tracking_uri': tracking_uri, 'mlflow_version': mlflow.__version__,
              'discovery': True}
    def empty_bundle(identity, name):
        return {'schema_version': SCHEMA_VERSION, 'source': {**source, 'seed_id': identity},
                'inventory': {}, 'dataset': {'name': name, 'records': []},
                'prompts': [], 'scorers': [], 'evaluations': [], 'traces': [], 'export_errors': []}
    primary = empty_bundle(source['seed_id'], 'trace-export')
    result = {'format': 'mlflow-discovery-v1', 'source': source, 'coverage': {},
              'bundles': [primary], 'export_errors': [],
              'limitations': [
                  'Evaluation/training run metadata is archived; historical experiment/example relationships are not reconstructed.',
                  'Each dataset remains separate. Dataset historical versions and arbitrary run artifacts are not exported.',
                  'Agent models, review workflow configuration, online schedules, serving and permissions are not recreated.',
                  'Prompt versions become separate private snapshots; aliases remain source metadata.',
                  'UC prompt discovery follows run/trace links and explicit names; unlinked registry prompts are outside this experiment export.',
                  'Discovery is bounded and refuses truncated results; it does not silently sample.']}
    def read(kind, fn):
        try:
            count = fn()
            result['coverage'][kind] = {'status': 'exported', 'count': count}
        except Exception as exc:
            error = {'resource': kind, 'error_type': type(exc).__name__}
            result['export_errors'].append(error)
            result['coverage'][kind] = {'status': 'failed', **error}

    def traces():
        primary['traces'] = export_traces(mlflow, experiment_id, limit)
        return len(primary['traces'])
    read('traces_and_assessments', traces)

    def datasets():
        rows = export_datasets(mlflow, experiment_id, tracking_uri, dataset_names, limit)
        for dataset in rows:
            identity = digest([source['seed_id'], 'dataset', dataset['name']])[:24]
            bundle = empty_bundle(identity, dataset['name'])
            bundle['dataset'] = dataset
            result['bundles'].append(bundle)
        return len(rows)
    read('datasets', datasets)

    def prompts():
        primary['prompts'] = export_prompts(mlflow, client, experiment_id, prompt_names, limit)
        return len(primary['prompts'])
    read('prompt_versions', prompts)

    def runs():
        rows = bounded(client.search_runs([str(experiment_id)], max_results=limit), limit, 'runs')
        result['source_runs'] = [plain(run) for run in rows]
        return len(rows)
    read('run_metadata', runs)

    def scorers():
        from mlflow.genai.scorers import list_scorers
        rows = list_scorers(experiment_id=str(experiment_id))
        primary['scorers'] = [{'name': s.name, 'sample_rate': s.sample_rate,
                               'filter_string': s.filter_string} for s in rows]
        return len(rows)
    read('scorer_configuration', scorers)
    primary['inventory'] = deepcopy(result['coverage'])
    for bundle in result['bundles']:
        bundle['export_errors'] = deepcopy(result['export_errors'])
    save_json(output_path, result)
    save_json(str(output_path) + '.coverage.json', {k: result[k] for k in ('coverage', 'export_errors', 'limitations')})
    if result['export_errors']:
        raise RuntimeError('Discovery incomplete. Inspect the saved coverage report; destination import is blocked.')
    return result
