from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

from helpers import bundle_fixture, FakeDestination, Recorder
from migration_core import plan_bundle, timestamp, convert_prompt
from migrate_mlflow import apply_plan, assert_run_matches
from seed_mlflow import build_agent
from stack_common import fixtures, PROMPTS, SCORERS
from evaluators import EVALUATORS


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.bundle = bundle_fixture()
        self.plan = plan_bundle(self.bundle)

    def test_replay_preserves_duration_hierarchy_and_original_dates(self):
        from migration_core import replay_plan, prepare_plans
        replay = replay_plan(self.plan, '2026-10-19T22:40:00+00:00')
        self.assertNotEqual(replay['bundle_digest'], self.plan['bundle_digest'])
        by_id = {r['id']: r for r in replay['runs']}
        for original, shifted in zip(self.plan['runs'], replay['runs']):
            duration = lambda r: datetime.fromisoformat(r['end_time']) - datetime.fromisoformat(r['start_time'])
            self.assertEqual(duration(original), duration(shifted))
            self.assertEqual(shifted['extra']['metadata']['source_start_time'], original['start_time'])
            self.assertEqual(shifted['inputs'], original['inputs'])
            if shifted['parent_run_id']:
                self.assertTrue(shifted['dotted_order'].startswith(by_id[shifted['parent_run_id']]['dotted_order'] + '.'))
        with TemporaryDirectory() as tmp:
            first = prepare_plans([self.bundle], 'test', tmp, True)
            self.assertEqual(first, prepare_plans([self.bundle], 'test', tmp, True))
            with self.assertRaises(ValueError): prepare_plans([self.bundle], 'test', tmp, False)

    def test_batch_upload_reuses_existing_runs_and_sets_project(self):
        from utils.mlflow_destination import LangSmithDestination
        destination = LangSmithDestination.__new__(LangSmithDestination)
        items = self.plan['runs'][:3]
        existing = SimpleNamespace(**{k:v for k,v in items[0].items() if k != 'project_name'})
        batches = []
        destination.client = SimpleNamespace(
            read_project=lambda **kw: SimpleNamespace(id='project-id'),
            list_runs=lambda **kw: [existing],
            batch_ingest_runs=lambda **kw: batches.append(kw['create']))
        completed = {}
        def perform(key, fn):
            if key not in completed: completed[key] = fn()
            return completed[key]
        destination.runs(items, perform)
        destination.runs(items, perform)
        self.assertEqual(len(batches), 1)
        self.assertEqual({r['id'] for r in batches[0]}, {r['id'] for r in items[1:]})
        self.assertTrue(all(r['session_id'] == 'project-id' for r in batches[0]))

    def test_seed_agent_improved_scores_and_actual_tools(self):
        recorder = Recorder()
        agent = build_agent(recorder)
        for row in fixtures():
            output = agent(row['id'], row['inputs']['messages'], 'improved', 'session', PROMPTS[1])
            actual_tools = [s['name'] for s in recorder.traces[-1]['spans'] if s['type'] == 'TOOL']
            self.assertEqual(actual_tools, output['tools'])
            self.assertTrue(all(fn(output, row['expectations']) for fn in SCORERS.values()))
            self.assertTrue(all(e(output, row['expectations'])['score'] == 1 for e in EVALUATORS))

    def test_baseline_has_real_failures_and_failed_root(self):
        baseline = self.bundle['evaluations'][0]
        self.assertLess(baseline['metrics']['answer_match'], 1)
        self.assertLess(baseline['metrics']['escalation_match'], 1)
        failed_roots = [r for r in self.plan['runs'] if r['error'] and r['parent_run_id'] is None]
        self.assertEqual(len(failed_roots), 1)

    def test_retry_error_and_nested_tools_preserved(self):
        runs = self.plan['runs']
        failed_tools = [r for r in runs if r['name'] == 'track_shipment' and r['error']]
        self.assertEqual(len(failed_tools), 2)
        for run in runs:
            if run['parent_run_id']:
                parent = next(r for r in runs if r['id'] == run['parent_run_id'])
                self.assertTrue(run['dotted_order'].startswith(parent['dotted_order'] + '.'))
                self.assertEqual(run['trace_id'], parent['trace_id'])
        nested_policy = [r for r in runs if r['name'] == 'search_policy' and
                         next(p for p in runs if p['id'] == r['parent_run_id'])['name'] == 'check_return_eligibility']
        self.assertEqual(len(nested_policy), 2)

    def test_stable_ids_and_source_not_mutated(self):
        before = deepcopy(self.bundle)
        self.assertEqual(self.plan, plan_bundle(self.bundle))
        self.assertEqual(before, self.bundle)
        self.assertEqual(len({r['id'] for r in self.plan['runs']}), len(self.plan['runs']))

    def test_prefix_isolates_destination_ids(self):
        other = plan_bundle(self.bundle, prefix='another-demo')
        self.assertTrue({r['id'] for r in self.plan['runs']}.isdisjoint({r['id'] for r in other['runs']}))
        self.assertTrue({r['id'] for r in self.plan['examples']}.isdisjoint({r['id'] for r in other['examples']}))

    def test_dataset_sdk_metadata_field_and_collision_detection(self):
        from migrate_mlflow import LangSmithDestination
        destination = LangSmithDestination.__new__(LangSmithDestination)
        class Missing(Exception):
            pass
        destination.not_found = Missing
        existing = SimpleNamespace(id='dataset-id', metadata={'migration_bundle_digest': self.plan['bundle_digest']})
        destination.client = SimpleNamespace(read_dataset=lambda **kwargs: existing)
        self.assertEqual(destination.dataset(self.plan), 'dataset-id')
        existing.metadata = {}
        with self.assertRaises(ValueError): destination.dataset(self.plan)

    def test_reference_link_and_thread_mapping(self):
        example_ids = {e['id'] for e in self.plan['examples']}
        for r in self.plan['runs']:
            self.assertTrue(r['extra']['metadata']['thread_id'])
            if r['parent_run_id'] is None:
                self.assertIn(r['reference_example_id'], example_ids)
            else:
                self.assertIsNone(r['reference_example_id'])

    def test_microsecond_precision_retains_original(self):
        first = self.plan['runs'][0]
        original = first['extra']['metadata']['source_start_ns']
        self.assertEqual(datetime.fromisoformat(first['start_time']).microsecond, original % 1_000_000_000 // 1000)
        self.assertRaises(ValueError, timestamp, None)

    def test_rejects_orphans_cycles_multiple_roots_and_incomplete(self):
        for kind in ('orphan', 'cycle', 'roots', 'incomplete', 'end_time'):
            bundle = deepcopy(self.bundle)
            spans = bundle['traces'][0]['spans']
            if kind == 'orphan': spans[1]['parent_id'] = 'missing'
            elif kind == 'cycle':
                spans[1]['parent_id'] = spans[2]['id']
                spans[2]['parent_id'] = spans[1]['id']
            elif kind == 'roots': spans[1]['parent_id'] = None
            elif kind == 'incomplete': bundle['export_errors'] = [{'trace_id': 'missing'}]
            else: spans[0]['end_ns'] = None
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                plan_bundle(bundle)

    def test_preserves_reference_not_prediction(self):
        row = self.bundle['dataset']['records'][0]
        row['outputs'] = {'answer': 'wrong observed answer'}
        plan = plan_bundle(self.bundle)
        self.assertEqual(plan['examples'][0]['outputs'], row['expectations'])

    def test_skipped_assessment_types_are_reported(self):
        assessments = self.bundle['traces'][0]['assessments']
        original = deepcopy(assessments[0])
        for changes in ({'kind': 'expectation'}, {'valid': False}, {'value': [1, 2]}, {'span_id': 'missing'}, {'id': None}):
            assessments.append({**original, **changes})
        plan = plan_bundle(self.bundle)
        self.assertEqual(len(plan['skipped']), 5)

    def test_prompt_conversion_is_conservative(self):
        self.assertEqual(convert_prompt('Hello {{ name }}'), 'Hello {name}')
        for template in ('{{ user.name }}', '{% if x %}', 'JSON {"x": 1}', [{'role': 'user', 'content': 'hi'}]):
            with self.assertRaises(ValueError): convert_prompt(template)

    def test_interruption_resume_and_no_duplicates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            destination = FakeDestination(fail_after=20)
            with self.assertRaises(ConnectionError): apply_plan(self.plan, destination, path)
            completed = len(json.loads(path.read_text())['completed'])
            self.assertEqual(completed, 20)
            destination.fail_after = None
            state = apply_plan(self.plan, destination, path, include_prompts=True)
            self.assertEqual(state['status'], 'applied_requires_validation')
            calls = destination.calls
            apply_plan(self.plan, destination, path, include_prompts=True)
            self.assertEqual(calls, destination.calls)
            self.assertEqual(len(state['completed']), len(destination.objects))
            destination.identity = 'another-workspace'
            with self.assertRaises(ValueError): apply_plan(self.plan, destination, path)

    def test_changed_bundle_cannot_reuse_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            destination = FakeDestination()
            apply_plan(self.plan, destination, path)
            self.bundle['dataset']['records'][0]['expectations']['answer'] = 'edited'
            with self.assertRaises(ValueError): apply_plan(plan_bundle(self.bundle), destination, path)

    def test_readback_detects_payload_and_timestamp_changes(self):
        expected = self.plan['runs'][0]
        actual = SimpleNamespace(**deepcopy(expected))
        assert_run_matches(actual, expected)
        actual.outputs = {'answer': 'corrupted'}
        with self.assertRaises(ValueError): assert_run_matches(actual, expected)
        actual = SimpleNamespace(**deepcopy(expected))
        actual.start_time = '2000-01-01T00:00:00+00:00'
        with self.assertRaises(ValueError): assert_run_matches(actual, expected)

class PortabilityTests(unittest.TestCase):
    def test_named_databricks_profiles(self):
        from stack_common import is_databricks_uri, databricks_registry_uri
        self.assertTrue(is_databricks_uri('databricks'))
        self.assertTrue(is_databricks_uri('databricks://trial'))
        self.assertFalse(is_databricks_uri('sqlite:///demo.db'))
        self.assertEqual(databricks_registry_uri('databricks://trial'), 'databricks-uc://trial')
        self.assertEqual(databricks_registry_uri('databricks'), 'databricks-uc')

    def test_local_export_downloads_evaluation_artifacts_with_profile(self):
        from unittest.mock import Mock, patch
        import sys
        from export_mlflow import export
        from stack_common import save_json
        fixture = bundle_fixture()
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            resources = {
                'seed/dataset.json': {'records': fixture['dataset']['records']},
                'seed/scorers.json': {'scorers': []},
                'seed/prompts.json': {'versions': fixture['prompts']},
                'seed/evaluations/evaluation-baseline.json': [{'case_id': 'billing-0', 'score': 1}],
            }
            for name, value in resources.items(): save_json(directory / 'remote' / name, value)
            manifest = {
                'schema_version': 1, 'seed_id': 'test', 'experiment_id': '123',
                'experiment_name': 'demo', 'tracking_uri': 'databricks', 'mlflow_version': '3.17.1',
                'resources': {}, 'dataset': fixture['dataset'], 'prompts': deepcopy(fixture['prompts']),
                'scorers': [], 'artifact_run_id': 'artifact-run', 'traces': [],
                'evaluations': [{'id': 'eval', 'status': 'created',
                    'results_file': 'evaluation-baseline.json',
                    'results_artifact': 'seed/evaluations/evaluation-baseline.json'}],
            }
            manifest['prompts'][0]['native'] = {'name': 'catalog.schema.prompt', 'version': '1'}
            save_json(directory / 'manifest.json', manifest)
            fake = SimpleNamespace(
                set_tracking_uri=Mock(), set_registry_uri=Mock(),
                get_experiment=Mock(return_value=SimpleNamespace(tags={}, trace_location=None)),
                artifacts=SimpleNamespace(download_artifacts=Mock(side_effect=lambda run_id, artifact_path: str(directory / 'remote' / artifact_path))),
                genai=SimpleNamespace(load_prompt=Mock(return_value=SimpleNamespace(template='Hello {{question}}'))),
            )
            with patch.dict(sys.modules, {'mlflow': fake}):
                result = export(directory / 'manifest.json', directory / 'bundle.json', 'databricks://trial')
            fake.set_tracking_uri.assert_called_once_with('databricks://trial')
            fake.set_registry_uri.assert_called_once_with('databricks-uc://trial')
            self.assertEqual(result['evaluations'][0]['rows'][0]['score'], 1)
            self.assertFalse((directory / 'evaluation-baseline.json').exists())


class WorkspacePathTests(unittest.TestCase):
    def test_conflicting_paths_rejected_before_sdk_or_filesystem_writes(self):
        from seed_mlflow import seed
        from unittest.mock import patch
        for output, experiment in (
            ('/Workspace/Users/demo/agent', '/Users/demo/agent'),
            ('/Workspace/Users/demo/agent/results', '/Users/demo/agent'),
            ('/Workspace/Users/demo/agent', '/Workspace/Users/demo/agent'),
            ('/Workspace/Users/demo/files/../agent', '/Users/demo/agent'),
        ):
            with self.subTest(output=output, experiment=experiment), patch.object(Path, 'mkdir') as mkdir:
                with self.assertRaisesRegex(ValueError, 'separate sibling paths'):
                    seed(output, experiment, tracking_uri='databricks://trial')
                mkdir.assert_not_called()

    def test_siblings_parent_output_and_local_output_are_allowed(self):
        from seed_mlflow import validate_source_paths
        for output, experiment in (
            ('/Workspace/Users/demo/agent-files', '/Users/demo/agent'),
            ('/Workspace/Users/demo', '/Users/demo/agent'),
            ('/tmp/agent', '/Users/demo/agent'),
        ):
            validate_source_paths(output, experiment, 'databricks')
        validate_source_paths('/Workspace/Users/demo/agent', '/Users/demo/agent', 'sqlite:///demo.db')


class EvaluationInventoryTests(unittest.TestCase):
    def test_persisted_eval_traces_replace_callback_assumptions(self):
        from unittest.mock import Mock
        from export_mlflow import reconcile_evaluation_links
        manifest = {'experiment_id': 'exp', 'dataset': {'records': [{'id': 'a'}, {'id': 'b'}]},
                    'traces': [{'trace_id': 'probe', 'evaluation': 'eval', 'case_id': 'a'}],
                    'evaluations': [{'id': 'eval', 'variant': 'baseline', 'run_id': 'run', 'status': 'created'}]}
        traces = [SimpleNamespace(info=SimpleNamespace(trace_id='trace-' + case, tags={'case_id': case},
                   trace_metadata={}), data=SimpleNamespace(spans=[])) for case in ('a', 'b')]
        sdk = SimpleNamespace(search_traces=Mock(return_value=traces))
        links, reports, errors = reconcile_evaluation_links(sdk, manifest)
        self.assertFalse(errors)
        sdk.search_traces.assert_called_once_with(locations=['exp'], run_id='run', return_type='list', max_results=10000)
        self.assertEqual(reports[0]['covered_cases'], 2)
        self.assertIsNone(links['probe']['evaluation'])
        self.assertEqual(links['probe']['evaluation_probe_for'], 'eval')
        self.assertEqual(links['trace-a']['evaluation'], 'eval')
        self.assertEqual(manifest['traces'][0]['evaluation'], 'eval')
        sdk.search_traces.return_value = traces[:1]
        _, _, errors = reconcile_evaluation_links(sdk, manifest)
        self.assertEqual(errors[0]['missing_cases'], ['b'])


if __name__ == '__main__':
    unittest.main()
