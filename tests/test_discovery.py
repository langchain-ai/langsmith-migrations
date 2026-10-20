import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import helpers
from discover_mlflow import bounded, export_experiment, list_experiments
from migration_core import plan_bundle
from migrate_mlflow import apply_plan


class DiscoveryTests(unittest.TestCase):
    def api(self, fail_datasets=False):
        def datasets(**kwargs):
            if fail_datasets:
                raise PermissionError('denied')
            return [NS(name=n, dataset_id=n, to_df=lambda: NS(to_dict=lambda **kwargs: [
                {'inputs': {'q': 'hello'}, 'expectations': {'answer': 'world'}}])) for n in ('one', 'two')]
        api = NS(__version__='test', search_traces=lambda **kwargs: [],
            genai=NS(datasets=NS(search_datasets=datasets),
                     search_prompts=lambda **kwargs: [NS(name='prompt-one'), NS(name='prompt-two')]))
        client = NS(get_experiment=lambda eid: NS(name='existing', tags={}),
            search_runs=lambda *args, **kwargs: [],
            search_model_versions=lambda **kwargs: [NS(version='1', aliases=[])],
            load_prompt=lambda uri: NS(template='Hello {{name}}'))
        return api, client

    def export(self, path, fail=False):
        with patch('discover_mlflow.connect', return_value=self.api(fail)), patch.dict('sys.modules', {
            'mlflow.genai.scorers': NS(list_scorers=lambda **kwargs: [])}):
            return export_experiment('exp', path, 'http://localhost', source_key='source')

    def test_discovery_preserves_datasets_and_unique_prompt_names_without_manifest(self):
        with TemporaryDirectory() as tmp:
            result = self.export(Path(tmp) / 'bundle.json')
            self.assertEqual(len(result['bundles']), 3)
            self.assertEqual(result['coverage']['datasets']['count'], 2)
            self.assertFalse(result['export_errors'])
            plans = [plan_bundle(b) for b in result['bundles']]
            self.assertFalse(plans[0]['create_dataset'])
            self.assertEqual(len({p['identifier'] for p in plans[0]['prompts']}), 2)
            self.assertNotEqual(plans[1]['dataset_name'], plans[2]['dataset_name'])
            self.assertFalse(plans[1]['projects'])
            state = apply_plan(plans[0], helpers.FakeDestination(), Path(tmp) / 'state.json')
            self.assertNotIn('dataset', state['completed'])

    def test_discovery_failure_saves_coverage_and_blocks_planning(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle.json'
            with self.assertRaisesRegex(RuntimeError, 'Discovery incomplete'):
                self.export(path, fail=True)
            result = json.loads(path.read_text())
            self.assertEqual(result['coverage']['datasets']['status'], 'failed')
            with self.assertRaisesRegex(ValueError, 'Incomplete source'):
                plan_bundle(result['bundles'][0])

    def test_uc_dataset_search_result_is_reloaded_by_name(self):
        from providers.mlflow.data.datasets import export_datasets
        found = NS(name='catalog.schema.data', dataset_id='search-id')
        loaded = NS(dataset_id='native-id', to_df=lambda: NS(to_dict=lambda **kw: [
            {'inputs': {'q': 'hi'}, 'expectations': {'answer': 'hello'}}]))
        calls = []
        def get_dataset(**kw):
            calls.append(kw)
            return loaded
        api = NS(genai=NS(datasets=NS(search_datasets=lambda **kw: [found], get_dataset=get_dataset)))
        result = export_datasets(api, '123', 'databricks://profile')
        self.assertEqual(calls, [{'name': 'catalog.schema.data'}])
        self.assertEqual(result[0]['native']['id'], 'native-id')
        self.assertEqual(len(result[0]['records']), 1)

    def test_uc_prompt_discovery_follows_links_without_search_filters(self):
        from providers.mlflow.data.prompts import export_prompts
        api = NS(get_registry_uri=lambda: 'databricks-uc://profile',
            search_traces=lambda **kw: [NS(info=NS(tags={'mlflow.linkedPrompts':
                '[{"name": "c.s.prompt", "version": "1"}]'}))])
        client = NS(search_runs=lambda *a, **kw: [NS(data=NS(tags={}))],
            search_prompt_versions=lambda name, **kw: [NS(version='1', aliases=['production'])],
            load_prompt=lambda uri: NS(template='Hello {{name}}'))
        result = export_prompts(api, client, '123')
        self.assertEqual(result, [{'name': 'c.s.prompt', 'version': '1',
            'template': 'Hello {{name}}', 'aliases': ['production']}])

    def test_bound_refuses_silent_truncation(self):
        with self.assertRaises(ValueError):
            bounded([1, 2], 2, 'traces')

    def test_experiment_listing_paginates(self):
        class Page(list):
            token = None
        first = Page([NS(experiment_id='1', name='one')]); first.token = 'next'
        second = Page([NS(experiment_id='2', name='two')])
        client = NS(search_experiments=lambda max_results, page_token: second if page_token else first)
        with patch('discover_mlflow.connect', return_value=(None, client)):
            self.assertEqual([e['id'] for e in list_experiments()], ['1', '2'])
