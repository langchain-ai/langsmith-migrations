"""In-memory tracing double. These records are NOT actual MLflow traces."""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seed_mlflow import build_agent
from stack_common import fixtures, PROMPTS, SCORERS


class Recorder:
    def __init__(self):
        self.stack = []
        self.traces = []
        self.clock = 1_760_000_000_123_456_789
        self.counter = 0

    @contextmanager
    def start_span(self, name, span_type):
        self.counter += 1
        self.clock += 1_000_000
        if not self.stack:
            self.current = {'id': 'trace-' + str(len(self.traces)), 'spans': [],
                            'metadata': {}, 'tags': {}, 'assessments': []}
            self.traces.append(self.current)
        span = {'id': f'span-{self.counter}', 'parent_id': self.stack[-1]['id'] if self.stack else None,
                'name': name, 'type': span_type, 'start_ns': self.clock, 'end_ns': None,
                'inputs': None, 'outputs': None, 'attributes': {}, 'status': 'OK', 'error': None}
        self.current['spans'].append(span)
        self.stack.append(span)
        obj = SimpleNamespace(
            set_inputs=lambda value: span.update(inputs=deepcopy(value)),
            set_outputs=lambda value: span.update(outputs=deepcopy(value)),
            set_attributes=lambda value: span['attributes'].update(deepcopy(value)))
        try:
            yield obj
        except Exception as exc:
            span.update(status='ERROR', error=str(exc))
            raise
        finally:
            self.clock += 1_000_000
            span['end_ns'] = self.clock
            self.stack.pop()

    def trace(self, span_type):
        def decorate(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                with self.start_span(function.__name__, span_type) as span:
                    span.set_inputs(dict(inspect.signature(function).bind(*args, **kwargs).arguments))
                    value = function(*args, **kwargs)
                    span.set_outputs(value)
                    return value
            return wrapped
        return decorate

    def update_current_trace(self, session_id, tags):
        self.current['session_id'] = session_id
        self.current['metadata']['mlflow.trace.session'] = session_id
        self.current['tags'].update(tags)


def bundle_fixture():
    recorder = Recorder()
    agent = build_agent(recorder)
    rows = fixtures()
    evaluations = []
    for variant in ('baseline', 'improved'):
        evaluation_id = 'eval-' + variant
        values = {name: [] for name in SCORERS}
        for row in rows:
            try:
                output = agent(row['id'], row['inputs']['messages'], variant,
                               f'session-{variant}-{row["metadata"]["scenario"]}',
                               PROMPTS[0 if variant == 'baseline' else 1])
            except TimeoutError:
                output = None
            trace = recorder.traces[-1]
            trace['link'] = {'case_id': row['id'], 'variant': variant, 'evaluation': evaluation_id}
            if output:
                for name, scorer in SCORERS.items():
                    value = scorer(output, row['expectations'])
                    values[name].append(value)
                    trace['assessments'].append({
                        'id': f'{trace["id"]}-{name}', 'kind': 'feedback', 'name': name, 'value': value,
                        'span_id': None, 'rationale': 'Offline fixture, not live MLflow evidence',
                        'source': {'source_type': 'CODE'}, 'metadata': {}, 'valid': True,
                        'overrides': None, 'error': None, 'raw': {}})
        evaluations.append({'id': evaluation_id, 'status': 'created', 'variant': variant,
                            'run_id': evaluation_id, 'metrics': {k: sum(v)/len(v) for k, v in values.items()}})
    return {'schema_version': 1,
            'source': {'seed_id': 'offline-fixture-001', 'experiment_id': 'fake-experiment',
                       'experiment_name': 'Offline fixture (not MLflow)', 'tracking_uri': 'fixture',
                       'mlflow_version': 'not-executed'},
            'inventory': {'offline_fixture': {'status': 'synthetic_test_only'}},
            'dataset': {'id': 'fake-dataset', 'name': 'support-cases', 'records': rows},
            'prompts': deepcopy(PROMPTS), 'scorers': [{'name': k, 'source': 'fixture'} for k in SCORERS],
            'evaluations': evaluations, 'traces': recorder.traces, 'export_errors': []}


class FakeDestination:
    identity = 'test-workspace'
    def __init__(self, fail_after=None):
        self.objects = {}
        self.calls = 0
        self.fail_after = fail_after
    def put(self, key, value):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ConnectionError('Injected network interruption')
        if key in self.objects:
            assert self.objects[key] == value
        self.objects[key] = value
        return key
    def dataset(self, plan): return self.put('dataset-id', plan['dataset_name'])
    def example(self, item, dataset_id): return self.put(item['id'], (item, dataset_id))
    def project(self, name, item, dataset_id, plan): return self.put(name, (item, dataset_id))
    def finalize(self, project_id): return self.put('finalized:' + project_id, project_id)
    def run(self, item): return self.put(item['id'], item)
    def feedback(self, item): return self.put(item['feedback_id'], item)
    def prompt(self, item): return self.put(item['identifier'], item)
