"""Export only resources identified by a seed manifest; never scan a whole workspace."""
from pathlib import Path
import argparse
import json
import os
import time
from copy import deepcopy
from stack_common import SCHEMA_VERSION, digest, plain, save_json, is_databricks_uri, databricks_registry_uri


def normalize_trace(trace, link):
    info = trace.info
    metadata = getattr(info, 'trace_metadata', None)
    if metadata is None:
        metadata = getattr(info, 'request_metadata', {})
    spans = []
    for span in trace.data.spans:
        status = str(span.status.status_code).split('.')[-1]
        spans.append({
            'id': span.span_id, 'parent_id': span.parent_id, 'name': span.name,
            'type': span.span_type, 'start_ns': span.start_time_ns, 'end_ns': span.end_time_ns,
            'inputs': plain(span.inputs), 'outputs': plain(span.outputs),
            'status': status, 'error': span.status.description if status == 'ERROR' else None,
            'attributes': plain(span.attributes),
        })
    assessments = []
    for a in info.assessments or []:
        kind = 'feedback' if a.feedback is not None else ('expectation' if a.expectation is not None else 'issue')
        assessments.append({
            'id': a.assessment_id, 'kind': kind, 'name': a.name,
            'value': plain(a.value) if kind in ('feedback', 'expectation') else None,
            'span_id': a.span_id, 'rationale': a.rationale,
            'source': plain(a.source), 'metadata': plain(a.metadata),
            'valid': a.valid, 'overrides': a.overrides,
            'error': plain(a.feedback.error) if kind == 'feedback' and a.feedback.error else None,
            'raw': plain(a),
        })
    return {'id': info.trace_id, 'metadata': plain(metadata), 'tags': plain(info.tags),
            'session_id': (metadata or {}).get('mlflow.trace.session') or link.get('session_id'),
            'link': deepcopy(link), 'spans': spans, 'assessments': assessments,
            'raw': plain(trace)}


def reconcile_evaluation_links(mlflow, manifest):
    """Prediction callbacks may run outside the driver; persisted runs are authoritative."""
    links = {entry['trace_id']: deepcopy(entry) for entry in manifest['traces']}
    expected_cases = {row['id'] for row in manifest['dataset']['records']}
    reports, errors = [], []
    for evaluation in manifest['evaluations']:
        if evaluation.get('status') != 'created' or not evaluation.get('run_id'):
            continue
        try:
            traces = mlflow.search_traces(
                locations=[manifest['experiment_id']], run_id=evaluation['run_id'],
                return_type='list', max_results=10000,
            )
            if len(traces) >= 10000:
                raise ValueError('Evaluation trace search reached its safety bound; pagination required')
            found = {}
            seen_cases = set()
            for trace in traces:
                candidates = set()
                tag_case = (trace.info.tags or {}).get('case_id')
                if tag_case in expected_cases:
                    candidates.add(tag_case)
                for span in trace.data.spans:
                    inputs = span.inputs
                    if isinstance(inputs, dict) and inputs.get('case_id') in expected_cases:
                        candidates.add(inputs['case_id'])
                if len(candidates) != 1:
                    raise ValueError('Cannot uniquely associate an evaluation trace with a dataset case')
                case_id = candidates.pop()
                metadata = getattr(trace.info, 'trace_metadata', None)
                if metadata is None:
                    metadata = getattr(trace.info, 'request_metadata', {})
                found[trace.info.trace_id] = {
                    'trace_id': trace.info.trace_id, 'case_id': case_id,
                    'variant': evaluation['variant'], 'evaluation': evaluation['id'],
                    'session_id': (metadata or {}).get('mlflow.trace.session'),
                }
                seen_cases.add(case_id)
            # Retain driver-side probes as ordinary traces, not evaluated predictions.
            for trace_id, entry in links.items():
                if entry.get('evaluation') == evaluation['id'] and trace_id not in found:
                    entry['evaluation_probe_for'] = entry.pop('evaluation')
                    entry['evaluation'] = None
            links.update(found)
            missing = sorted(expected_cases - seen_cases)
            reports.append({'evaluation_id': evaluation['id'], 'run_id': evaluation['run_id'],
                            'persisted_trace_count': len(found), 'covered_cases': len(seen_cases),
                            'expected_cases': len(expected_cases), 'missing_cases': missing})
            if missing:
                errors.append({'evaluation_id': evaluation['id'], 'error_type': 'MissingEvaluationCases',
                               'missing_cases': missing})
        except Exception as exc:
            errors.append({'evaluation_id': evaluation['id'], 'error_type': type(exc).__name__})
    return links, reports, errors


def discover_experiment_traces(mlflow, manifest, links):
    """Include later manual/monitoring traces within this explicitly selected experiment."""
    found = mlflow.search_traces(locations=[manifest['experiment_id']],
                                return_type='list', max_results=10000)
    if len(found) >= 10000:
        raise ValueError('Experiment trace discovery reached 10000; pagination required before export.')
    for trace in found:
        tags = trace.info.tags or {}
        metadata = trace.info.trace_metadata or {}
        links.setdefault(trace.info.trace_id, {
            'trace_id': trace.info.trace_id, 'case_id': tags.get('case_id'),
            'variant': tags.get('variant'), 'evaluation': None,
            'session_id': metadata.get('mlflow.trace.session'), 'discovered': True})
    return len(found)


def export(manifest_path, output_path, tracking_uri=None, discover_traces=False):
    import mlflow
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported manifest schema')
    effective_tracking_uri = tracking_uri or manifest['tracking_uri']
    mlflow.set_tracking_uri(effective_tracking_uri)
    if is_databricks_uri(effective_tracking_uri):
        experiment = mlflow.get_experiment(manifest['experiment_id'])
        warehouse = os.environ.get('MLFLOW_TRACING_SQL_WAREHOUSE_ID') or (experiment.tags or {}).get('mlflow.monitoring.sqlWarehouseId')
        if warehouse:
            os.environ['MLFLOW_TRACING_SQL_WAREHOUSE_ID'] = warehouse
        elif getattr(experiment, 'trace_location', None) is not None:
            raise ValueError('UC export requires MLFLOW_TRACING_SQL_WAREHOUSE_ID or the experiment monitoring warehouse tag.')
    bundle = {'schema_version': SCHEMA_VERSION,
              'source': {k: manifest[k] for k in ('seed_id', 'experiment_id', 'experiment_name', 'tracking_uri', 'mlflow_version')},
              'inventory': deepcopy(manifest['resources']), 'dataset': deepcopy(manifest['dataset']),
              'prompts': deepcopy(manifest['prompts']), 'scorers': deepcopy(manifest['scorers']),
              'evaluations': deepcopy(manifest['evaluations']), 'traces': [], 'export_errors': []}
    bundle['source_extensions'] = {k: deepcopy(manifest[k]) for k in
        ('agent_versions', 'review_session', 'previous_evaluations', 'extension', 'tab_report') if k in manifest}
    def artifact(name):
        local = mlflow.artifacts.download_artifacts(run_id=manifest['artifact_run_id'], artifact_path='seed/' + name)
        return json.loads(Path(local).read_text())
    # Read persisted artifacts rather than trusting in-memory seed fixtures.
    dataset_artifact = artifact('dataset.json')['records']
    bundle['dataset']['records'] = dataset_artifact
    bundle['scorers'] = artifact('scorers.json')['scorers']
    native = bundle['dataset'].get('native')
    if native:
        ds = mlflow.genai.datasets.get_dataset(name=native['name']) if is_databricks_uri(effective_tracking_uri) else mlflow.genai.datasets.get_dataset(dataset_id=native['id'])
        native_rows = plain(ds.to_df().to_dict(orient='records'))
        by_inputs = {digest(r['inputs']): r for r in dataset_artifact}
        merged = []
        for r in native_rows:
            key = digest(r['inputs'])
            if key not in by_inputs:
                raise ValueError('Native dataset has an unmanifested row; expand source manifest explicitly.')
            base = deepcopy(by_inputs[key])
            base['inputs'], base['expectations'] = r['inputs'], r['expectations']
            base['native_record'] = r
            merged.append(base)
        if len(merged) != len(dataset_artifact):
            raise ValueError('Native dataset count differs from the seed artifact; review before export.')
        bundle['dataset']['records'] = merged
    persisted_prompts = {p['version']: p for p in artifact('prompts.json')['versions']}
    for p in bundle['prompts']:
        if p.get('native'):
            if is_databricks_uri(effective_tracking_uri):
                mlflow.set_registry_uri(databricks_registry_uri(effective_tracking_uri))
            native = p['native']
            pv = mlflow.genai.load_prompt(f"prompts:/{native['name']}/{native['version']}")
            p['template'] = plain(pv.template)
        else:
            p['template'] = persisted_prompts[p['version']]['template']
    for e in bundle['evaluations']:
        if e.get('run_id'):
            run = mlflow.MlflowClient().get_run(e['run_id'])
            e['metrics'] = plain(run.data.metrics)
            e['tags'] = plain(run.data.tags)
        if e.get('results_artifact'):
            local = mlflow.artifacts.download_artifacts(
                run_id=manifest['artifact_run_id'], artifact_path=e['results_artifact'])
            e['rows'] = json.loads(Path(local).read_text())
        elif e.get('results_file'):
            local = manifest_path.parent / e['results_file']
            if not local.is_file():
                raise FileNotFoundError('Older seed manifest: copy its evaluation-*.json files alongside the manifest, or regenerate with the current seed.')
            e['rows'] = json.loads(local.read_text())
    links, reports, errors = reconcile_evaluation_links(mlflow, manifest)
    if discover_traces:
        bundle['discovered_trace_count'] = discover_experiment_traces(mlflow, manifest, links)
        from mlflow.genai.scorers import list_scorers
        bundle['registered_scorers'] = [
            {'name': s.name, 'sample_rate': s.sample_rate, 'filter_string': s.filter_string}
            for s in list_scorers(experiment_id=manifest['experiment_id'])]
    bundle['evaluation_trace_reconciliation'] = reports
    bundle['export_errors'].extend(errors)
    for trace_id, link in links.items():
        try:
            trace = None
            for attempt in range(6):
                trace = mlflow.get_trace(trace_id)
                if trace is not None:
                    break
                time.sleep(2)
            if trace is None:
                raise LookupError('Trace not available')
            bundle['traces'].append(normalize_trace(trace, link))
        except Exception as exc:
            bundle['export_errors'].append({'trace_id': trace_id, 'error_type': type(exc).__name__})
    save_json(output_path, bundle)
    print(f'Exported {len(bundle["traces"])} traces; {len(bundle["export_errors"])} errors. Bundle: {output_path}')
    if bundle['export_errors']:
        raise RuntimeError('Export incomplete. Migration refuses this bundle; retry export after resolving errors.')
    return bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tracking-uri', help='Override source authentication routing, e.g. databricks://PROFILE')
    parser.add_argument('--discover-traces', action='store_true', help='Include all traces in the selected experiment (bounded at 10000).')
    args = parser.parse_args()
    export(args.manifest, args.output, args.tracking_uri, args.discover_traces)
