"""Verify partial backend support cannot be reported as a complete stack."""
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import helpers  # Adds the source directory to sys.path.
from extend_mlflow import trace_has_model, review_input, ensure_agent_records, ensure_review_schema, configure_warehouse, setup, extend, validate_namespace, discover_namespaces, response_present, successful_monitor_assessments


class ExtensionTests(TestCase):
    def test_monitoring_warehouse_persists_for_remote_job_on_older_sdk(self):
        from extend_mlflow import configure_monitoring_warehouse
        tags = {}
        client = SimpleNamespace(set_experiment_tag=lambda eid, key, value: tags.update({key: value}))
        api = SimpleNamespace(tracing=SimpleNamespace(), MlflowClient=lambda: client,
                              get_experiment=lambda eid: SimpleNamespace(tags=tags))
        configure_monitoring_warehouse(api, 'exp', 'warehouse')
        self.assertEqual(tags, {'mlflow.monitoring.sqlWarehouseId': 'warehouse'})

    def test_uc_trace_metadata_verifies_exact_model_link(self):
        trace = SimpleNamespace(info=SimpleNamespace(trace_metadata={'mlflow.modelId': 'model-1'}))
        self.assertTrue(trace_has_model(trace, 'model-1'))
        self.assertFalse(trace_has_model(trace, 'model-2'))
        self.assertFalse(trace_has_model(None, 'model-1'))

    def test_review_input_uses_installed_method_annotation(self):
        class InputCategorical:
            def __init__(self, options):
                self.options = options
        class App:
            def create_label_schema(self, input: InputCategorical):
                pass
        self.assertEqual(review_input(App()).options, ['yes', 'no'])

    def test_model_creation_resumes_missing_variant_before_trace_checks(self):
        created, checkpoints = [], []
        def create(**kw):
            created.append(kw['tags']['variant'])
            return SimpleNamespace(model_id='improved-id')
        manifest = {'seed_id': 'seed', 'experiment_id': 'exp',
                    'agent_versions': {'baseline': {'model_id': 'baseline-id'}}}
        api = SimpleNamespace(create_external_model=create)
        ensure_agent_records(api, manifest, lambda: checkpoints.append(True))
        ensure_agent_records(api, manifest, lambda: checkpoints.append(True))
        self.assertEqual(created, ['improved'])
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(set(manifest['agent_versions']), {'baseline', 'improved'})

    def test_review_schema_is_created_once_and_returns_nonempty_name(self):
        calls = []
        app = SimpleNamespace(label_schemas=[])
        def create(**kw):
            calls.append(kw)
            app.label_schemas.append(SimpleNamespace(name=kw['name']))
        app.create_label_schema = create
        self.assertEqual(ensure_review_schema(app, 'categorical'), 'synthetic_support_quality')
        ensure_review_schema(app, 'categorical')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['type'], 'feedback')

    def test_warehouse_discovery_prefers_running_and_preserves_override(self):
        ws = SimpleNamespace(warehouses=SimpleNamespace(list=lambda: [
            SimpleNamespace(id='stopped', name='Stopped', state='STOPPED'),
            SimpleNamespace(id='running', name='Running', state='RUNNING')]))
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(configure_warehouse('databricks', ws), 'running')
            # No second lookup if explicitly configured or already selected.
            self.assertEqual(configure_warehouse('databricks', object()), 'running')

    def test_no_visible_warehouse_stops_preflight(self):
        ws = SimpleNamespace(warehouses=SimpleNamespace(list=lambda: []))
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'No SQL warehouse'):
                configure_warehouse('databricks', ws)

    def test_storage_creation_failure_prevents_any_seeding(self):
        def fail(*args, **kwargs):
            raise RuntimeError('UC unavailable')
        fake = SimpleNamespace(set_tracking_uri=lambda uri: None,
            get_experiment=lambda eid: SimpleNamespace(trace_location=None),
            get_experiment_by_name=lambda name: None, create_experiment=fail)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({'tracking_uri': 'databricks', 'experiment_id': '123',
                'experiment_name': '/Users/demo/source',
                'dataset': {'native': {'name': 'workspace.default.cases'}}}))
            before = path.read_text()
            with patch.dict('sys.modules', {'mlflow': fake,
                    'mlflow.entities.trace_location': SimpleNamespace(UnityCatalog=lambda **kw: kw)}):
                with patch('seed_mlflow.seed') as seed_mock, patch('extend_mlflow.configure_warehouse'):
                    with self.assertRaisesRegex(RuntimeError, 'UC unavailable'):
                        setup(path)
                    seed_mock.assert_not_called()
            self.assertEqual(path.read_text(), before)

    def test_online_scorer_requires_observed_answer_not_ground_truth(self):
        self.assertTrue(response_present({'answer': 'Friday'}))
        for outputs in (None, {}, {'answer': ''}, {'answer': '   '}, 'Friday'):
            self.assertFalse(response_present(outputs))

    def test_failed_or_invalid_assessments_do_not_verify_monitoring(self):
        def assessment(name='monitor', error=None, valid=True):
            return SimpleNamespace(name=name, feedback=SimpleNamespace(error=error), valid=valid)
        good = assessment()
        trace = SimpleNamespace(info=SimpleNamespace(assessments=[
            assessment(error='timeout'), assessment(valid=False), assessment(name='offline'), good]))
        self.assertEqual(successful_monitor_assessments(trace, 'monitor'), [good])

    def test_discovery_prefers_current_and_excludes_system_locations(self):
        queries = []
        def sql(query):
            queries.append(query)
            rows = {
                'SELECT current_catalog(), current_schema()': [('workspace', 'demo')],
                'SHOW CATALOGS': [('other',), ('workspace',), ('system',), ('hive_metastore',), ('samples',)],
                'SHOW SCHEMAS IN `workspace`': [('default',), ('demo',), ('information_schema',)],
                'SHOW SCHEMAS IN `other`': [('default',)],
            }[query]
            return SimpleNamespace(first=lambda: rows[0], collect=lambda: rows)
        self.assertEqual(discover_namespaces(SimpleNamespace(sql=sql)),
                         ['workspace.demo', 'workspace.default', 'other.default'])
        self.assertTrue(all(q.startswith(('SELECT ', 'SHOW ')) for q in queries))

    def test_namespace_validation(self):
        validate_namespace(None)
        validate_namespace('demo.agent_stack')
        for value in ('', 'catalog', 'catalog.schema.table', 'catalog.schema; DROP TABLE x'):
            with self.assertRaises(ValueError):
                validate_namespace(value)

    def test_partial_backend_is_reported_and_inventory_synced(self):
        def unavailable(*args, **kwargs):
            raise NotImplementedError('SDK capability unavailable')
        uploads = {}
        fake = SimpleNamespace(
            set_tracking_uri=lambda uri: None, set_registry_uri=lambda uri: None,
            set_experiment=lambda **kw: None, MlflowClient=lambda: SimpleNamespace(),
            genai=SimpleNamespace(get_labeling_sessions=unavailable),
            trace=lambda **kw: lambda fn: fn,
            create_external_model=unavailable, search_traces=unavailable,
            start_run=lambda **kw: nullcontext(),
            log_dict=lambda value, name: uploads.update({name: value}))
        scorers = SimpleNamespace(scorer=unavailable, list_scorers=unavailable)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({
                'tracking_uri': 'databricks', 'experiment_id': '123', 'seed_id': 'demo',
                'artifact_run_id': 'inventory', 'resources': {}, 'traces': [],
                'evaluations': [], 'dataset': {'records': [{'id': 'delivery-0'}]},
                'prompts': [{'version': '1'}]}))
            with patch.dict('sys.modules', {'mlflow': fake, 'mlflow.genai.scorers': scorers}):
                result = extend(path)
            report = result['tab_report']
            self.assertEqual(report['Datasets']['status'], 'blocked')
            self.assertEqual(report['Prompts']['status'], 'blocked')
            for tab in ('Scorers', 'Agent versions', 'Reviews', 'Traces', 'Sessions', 'Evaluation runs'):
                self.assertEqual(report[tab]['status'], 'failed')
            self.assertEqual(uploads['seed/manifest.json'], json.loads(path.read_text()))
            self.assertEqual(uploads['seed/tab-report.json'], report)
            self.assertFalse(result['setup_complete'])
            self.assertEqual(report['UC trace storage']['status'], 'failed')
            self.assertEqual(report['Native evaluations']['status'], 'blocked')
            self.assertEqual(report['Monitoring traffic']['status'], 'blocked')
