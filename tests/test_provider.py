import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from helpers import bundle_fixture
from export_mlflow import discover_experiment_traces


class ProviderTests(unittest.TestCase):
    def test_local_bundle_notebook_runs_preview_without_source_or_destination_sdks(self):
        root = Path(__file__).resolve().parents[1]
        notebook = root / 'providers/mlflow/03_databricks_export_migrate.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        markdown = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'markdown')
        for anchor in ('scenario-1', 'scenario-2', 'configure', 'export', 'preview', 'import', 'continue'):
            self.assertIn(f'#{anchor}', markdown)
            self.assertIn(f'id="{anchor}"', markdown)
        source = '\n\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
        source = source.replace('EXECUTION_ENVIRONMENT = "auto"', 'EXECUTION_ENVIRONMENT = "local"')
        source = source.replace('EXPORT_SOURCE = True', 'EXPORT_SOURCE = False')
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'migration-export').mkdir()
            (tmp / 'migration-export/bundle.json').write_text(json.dumps(bundle_fixture()))
            script = tmp / 'run_notebook.py'
            script.write_text(source)
            result = subprocess.run([sys.executable, str(script)], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((tmp / 'migration-export/export-summary.json').read_text())
            self.assertFalse(summary['destination_writes'])
            self.assertFalse((tmp / 'migration-export/migration-state.json').exists())

    def test_discovery_keeps_evaluation_links_and_adds_monitoring_trace(self):
        links = {'eval': {'evaluation': 'baseline'}}
        traces = [SimpleNamespace(info=SimpleNamespace(trace_id=i, tags={}, trace_metadata={}))
                  for i in ('eval', 'later-monitor')]
        api = SimpleNamespace(search_traces=lambda **kwargs: traces)
        self.assertEqual(discover_experiment_traces(api, {'experiment_id': 'exp'}, links), 2)
        self.assertEqual(links['eval']['evaluation'], 'baseline')
        self.assertIsNone(links['later-monitor']['evaluation'])
        self.assertTrue(links['later-monitor']['discovered'])

    def test_root_dispatcher_previews_without_other_provider_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'bundle.json').write_text(json.dumps(bundle_fixture()))
            result = subprocess.run([sys.executable, str(root / 'migrate.py'),
                '--provider', 'mlflow', '--bundle', str(tmp / 'bundle.json'),
                '--output', str(tmp / 'out')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((tmp / 'out/migration-summary.json').read_text())['mode'], 'preview')

    def test_final_runbook_offline_cells_execute(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'providers/mlflow/runbook.ipynb').read_text())
        code = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Offline validation passed', result.stdout)

    def test_provider_previews_exported_bundle_without_service_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'bundle.json').write_text(json.dumps(bundle_fixture()))
            result = subprocess.run([sys.executable, str(root / 'providers/mlflow/main.py'),
                '--bundle', str(tmp / 'bundle.json'), '--output', str(tmp / 'out')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((tmp / 'out/migration-summary.json').read_text())
            self.assertEqual(summary['mode'], 'preview')
            self.assertGreater(summary['counts']['runs'], 0)
            self.assertFalse((tmp / 'out/migration-state.json').exists())
