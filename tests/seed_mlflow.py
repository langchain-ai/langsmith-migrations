"""Create a representative, synthetic MLflow source stack (manual Databricks execution)."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4
import argparse
import inspect
import json
import posixpath
import time

from stack_common import SCHEMA_VERSION, SCORERS, PROMPTS, fixtures, plain, save_json, is_databricks_uri, databricks_registry_uri


def build_agent(mlflow):
    """Scripted orchestration with actual traced Python tools, not a real LLM."""
    @mlflow.trace(span_type='TOOL')
    def search_policy(topic):
        return {'topic': topic, 'return_days': 30, 'opened_allowed': True, 'hold_days': '3–5'}

    @mlflow.trace(span_type='TOOL')
    def check_return_eligibility(days):
        return days <= search_policy('returns')['return_days']

    @mlflow.trace(span_type='TOOL')
    def lookup_order(order_id):
        return {'order_id': order_id, 'tracking_id': 'DEMO-TRACK'}

    @mlflow.trace(span_type='TOOL')
    def track_shipment(tracking_id, fail=False):
        if fail:
            raise TimeoutError('Synthetic carrier timeout')
        return {'eta': 'Friday', 'location': 'local depot', 'signature': False}

    @mlflow.trace(span_type='TOOL')
    def lookup_payments():
        return {'posted': 1, 'pending': 1}

    @mlflow.trace(span_type='TOOL')
    def create_demo_ticket(category):
        return {'id': 'FAKE-' + uuid4().hex[:8], 'category': category, 'simulated': True}

    def agent(case_id, messages, variant, session_id, prompt):
        scenario, turn = case_id.rsplit('-', 1)
        turn = int(turn)
        tools = []
        escalated = False
        with mlflow.start_span(name='support_agent', span_type='AGENT') as root:
            root.set_inputs({'case_id': case_id, 'messages': deepcopy(messages)})
            mlflow.update_current_trace(session_id=session_id, tags={
                'synthetic': 'true', 'variant': variant, 'case_id': case_id,
                'prompt_version': str(prompt['version']), 'scenario': scenario,
            })
            facts = {}
            def call(name, fn, *args, **kwargs):
                tools.append(name)
                return fn(*args, **kwargs)
            with mlflow.start_span(name='execute_plan', span_type='CHAIN') as plan:
                plan.set_inputs({'scenario': scenario, 'turn': turn, 'variant': variant})
                if scenario == 'delivery':
                    if turn == 0:
                        order = call('lookup_order', lookup_order, 'DEMO-101')
                        facts = call('track_shipment', track_shipment, order['tracking_id'])
                        answer = f"Your order is expected {facts['eta']}."
                    else:
                        facts = call('track_shipment', track_shipment, 'DEMO-TRACK')
                        answer = (f"Yes, delivery is expected {facts['eta']}." if turn == 1
                                  else 'No signature is required.')
                elif scenario == 'returns':
                    if turn == 0:
                        eligible = call('check_return_eligibility', check_return_eligibility, 12)
                        tools.append('search_policy')  # The nested tool also executes above.
                        facts = {'eligible': eligible}
                        answer = 'Yes, the return window is 30 days.'
                    elif turn == 1:
                        facts = call('search_policy', search_policy, 'returns')
                        answer = ('Opened items are ineligible.' if variant == 'baseline'
                                  else 'Opened items are eligible.')
                    else:
                        facts = call('create_demo_ticket', create_demo_ticket, 'return')
                        escalated = True
                        answer = 'A simulated return ticket was created; no real action was taken.'
                elif scenario == 'billing':
                    if turn < 2:
                        facts = call('lookup_payments', lookup_payments)
                        if turn == 0:
                            facts['policy'] = call('search_policy', search_policy, 'billing')
                        answer = ('One charge is posted and one is pending.' if turn == 0
                                  else 'Only one charge has posted.')
                        if variant == 'baseline' and turn == 1:
                            answer = 'You paid twice.'
                    elif variant == 'baseline':
                        answer = 'Please wait longer.'
                    else:
                        facts = call('create_demo_ticket', create_demo_ticket, 'billing')
                        escalated = True
                        answer = 'A simulated billing ticket was created; no real action was taken.'
                elif scenario == 'retry':
                    if turn == 0:
                        order = call('lookup_order', lookup_order, 'DEMO-202')
                        try:
                            facts = call('track_shipment', track_shipment, order['tracking_id'], fail=True)
                        except TimeoutError:
                            if variant == 'baseline':
                                raise
                            facts = call('track_shipment', track_shipment, order['tracking_id'])
                        answer = f"The retry succeeded; delivery is expected {facts['eta']}."
                    elif turn == 1:
                        facts = call('track_shipment', track_shipment, 'DEMO-TRACK')
                        answer = f"Your package is at the {facts['location']}."
                    else:
                        facts = call('create_demo_ticket', create_demo_ticket, 'delivery')
                        escalated = True
                        answer = 'A simulated delivery ticket was created; no real action was taken.'
                else:
                    raise ValueError('Unknown demo scenario')
                plan.set_outputs({'facts': facts, 'tools': tools})
            with mlflow.start_span(name='scripted_model', span_type='CHAT_MODEL') as model:
                model.set_attributes({'demo.synthetic': True, 'demo.model': 'scripted-fixture'})
                if prompt.get('native'):
                    native = prompt['native']
                    loaded_prompt = mlflow.genai.load_prompt(f"prompts:/{native['name']}/{native['version']}")
                    rendered_prompt = loaded_prompt.format(question=messages[-1]['content'])
                else:
                    rendered_prompt = prompt['template'].replace('{{question}}', messages[-1]['content'])
                model.set_inputs({'messages': deepcopy(messages), 'facts': facts,
                                  'rendered_prompt': rendered_prompt})
                model.set_outputs({'messages': [{'role': 'assistant', 'content': answer}]})
            output = {'answer': answer, 'tools': tools, 'escalated': escalated,
                      'messages': [{'role': 'assistant', 'content': answer}]}
            root.set_outputs(output)
            return output
    return agent


def validate_source_paths(output_dir, experiment_name, tracking_uri):
    """Workspace directories and experiments share one Databricks namespace."""
    if not is_databricks_uri(tracking_uri):
        return
    output = posixpath.normpath(str(Path(output_dir).absolute()))
    if not output.startswith('/Workspace/'):
        return  # Local files outside the workspace mount do not reserve workspace nodes.
    output_node = PurePosixPath(output.removeprefix('/Workspace'))
    experiment = posixpath.normpath(experiment_name)
    if experiment.startswith('/Workspace/'):
        experiment = experiment.removeprefix('/Workspace')
    experiment_node = PurePosixPath(experiment)
    if output_node == experiment_node or experiment_node in output_node.parents:
        raise ValueError(
            'OUTPUT_DIR must not equal or be inside the Databricks experiment path. '
            'Use separate sibling paths, e.g. /Users/you/demo-experiment and '
            '/Workspace/Users/you/demo-files. No directories or experiments were created.'
        )


def seed(output_dir, experiment_name, tracking_uri='databricks', uc_prefix=None):
    validate_source_paths(output_dir, experiment_name, tracking_uri)
    output_dir = Path(output_dir)
    if (output_dir / 'manifest.json').exists():
        raise FileExistsError('Choose a fresh output directory for each seed execution.')
    import mlflow
    from mlflow.genai.scorers import scorer
    from mlflow.entities import AssessmentSource, AssessmentSourceType
    if int(mlflow.__version__.split('.')[0]) < 3:
        raise RuntimeError('MLflow 3.x is required; reference API version is 3.17.1.')
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    seed_id = uuid4().hex
    rows = fixtures()
    manifest = {
        'schema_version': SCHEMA_VERSION, 'seed_id': seed_id,
        'mlflow_version': mlflow.__version__, 'tracking_uri': tracking_uri,
        'experiment_id': experiment.experiment_id, 'experiment_name': experiment_name,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'resources': {}, 'traces': [], 'evaluations': [], 'scorers': [],
        'dataset': {'id': 'support-cases-' + seed_id, 'name': 'support-cases', 'records': rows},
        'prompts': [{'name': 'support-assistant', **deepcopy(p)} for p in PROMPTS],
    }
    def checkpoint():
        save_json(output_dir / 'manifest.json', manifest)
    def attempt(name, fn):
        try:
            result = fn()
            manifest['resources'][name] = {'status': 'created', 'result': plain(result)}
            checkpoint()
            return result
        except Exception as exc:
            # Do not echo authentication-bearing URLs or vendor exception payloads.
            manifest['resources'][name] = {'status': 'failed', 'error_type': type(exc).__name__}
            checkpoint()
            print(f'{name}: {type(exc).__name__}; inspect permissions/backend support locally.')
            return None
    checkpoint()
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='agent-stack-inventory') as run:
        manifest['artifact_run_id'] = run.info.run_id
        mlflow.log_dict({'records': rows}, 'seed/dataset.json')
        mlflow.log_dict({'versions': PROMPTS}, 'seed/prompts.json')
        manifest['resources']['dataset_artifact'] = {'status': 'created', 'run_id': run.info.run_id}
        manifest['resources']['prompt_artifact'] = {'status': 'created', 'run_id': run.info.run_id}
    native_evaluation_dataset = None
    if uc_prefix or not is_databricks_uri(tracking_uri):
        dataset_name = f'{uc_prefix}.support_cases_{seed_id[:8]}' if uc_prefix else f'support_cases_{seed_id[:8]}'
        def create_native_dataset():
            nonlocal native_evaluation_dataset
            ds = mlflow.genai.datasets.create_dataset(name=dataset_name, experiment_id=experiment.experiment_id)
            ds.merge_records([{'inputs': r['inputs'], 'expectations': r['expectations']} for r in rows])
            native_evaluation_dataset = ds
            return {'name': ds.name, 'id': ds.dataset_id}
        native_dataset = attempt('native_dataset', create_native_dataset)
        if native_dataset:
            manifest['dataset']['native'] = native_dataset
        if is_databricks_uri(tracking_uri):
            mlflow.set_registry_uri(databricks_registry_uri(tracking_uri))
        prompt_name = f'{uc_prefix}.support_prompt_{seed_id[:8]}' if uc_prefix else f'support_prompt_{seed_id[:8]}'
        for prompt in manifest['prompts']:
            def register(p=prompt):
                pv = mlflow.genai.register_prompt(name=prompt_name, template=p['template'])
                return {'name': pv.name, 'version': str(pv.version)}
            native = attempt('prompt_version_' + prompt['version'], register)
            if native:
                prompt['native'] = native
                for alias in prompt['aliases']:
                    attempt('prompt_alias_' + alias, lambda n=native, a=alias: (
                        mlflow.genai.set_prompt_alias(n['name'], a, int(n['version'])), {'alias': a})[1])
    else:
        manifest['resources']['native_dataset'] = {'status': 'not_requested', 'reason': 'Set UC_PREFIX=catalog.schema to create a native Databricks dataset.'}
        manifest['resources']['native_prompts'] = {'status': 'not_requested', 'reason': 'Set UC_PREFIX=catalog.schema to register native prompts.'}
    scorers = []
    for name, function in SCORERS.items():
        decorated = scorer(name=name)(function)
        scorers.append(decorated)
        manifest['scorers'].append({'name': name, 'kind': 'custom_python', 'source': inspect.getsource(function), 'function_name': function.__name__})
        attempt('registered_scorer_' + name, lambda s=decorated: {
            'name': s.register(experiment_id=experiment.experiment_id).name})
    agent = build_agent(mlflow)
    def capture(case_id, messages, variant, session_id, evaluation=None):
        try:
            return agent(case_id, messages, variant, session_id, manifest['prompts'][0 if variant == 'baseline' else 1])
        finally:
            active = mlflow.get_current_active_span()
            trace_id = active.trace_id if active else mlflow.get_last_active_trace_id()
            if trace_id:
                manifest['traces'].append({'trace_id': trace_id, 'case_id': case_id, 'variant': variant,
                                           'session_id': session_id, 'evaluation': evaluation})
    # Continuous conversations: history uses the previous observed response, not ground truth.
    for variant in ('baseline', 'improved'):
        histories = {}
        for row in rows:
            scenario = row['metadata']['scenario']
            history = histories.setdefault(scenario, [deepcopy(row['inputs']['messages'][0])])
            history.append(deepcopy(row['inputs']['messages'][-1]))
            try:
                output = capture(row['id'], history, variant, f'{seed_id}-{variant}-{scenario}')
                history.append(output['messages'][0])
            except TimeoutError:
                history.append({'role': 'assistant', 'content': 'The demo carrier is unavailable.'})
            checkpoint()
    mlflow.flush_trace_async_logging()
    # A synthetic human-source assessment; never claim this was a real reviewer.
    for entry in manifest['traces'][:2]:
        attempt('assessment_' + entry['trace_id'], lambda e=entry: mlflow.log_feedback(
            trace_id=e['trace_id'], name='demo_human_review', value=True,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id='synthetic-reviewer'),
            rationale='Synthetic review fixture; not a real human judgment.', metadata={'synthetic': True}).to_dictionary())
    # Offline evals use fixed input histories to compare variants on identical cases.
    for variant in ('baseline', 'improved'):
        evaluation = {'id': f'{seed_id}-{variant}', 'variant': variant, 'status': 'running',
                      'dataset_storage': 'native' if native_evaluation_dataset is not None else 'artifact_fixture'}
        manifest['evaluations'].append(evaluation)
        def predict_fn(case_id, messages):
            return capture(case_id, messages, variant, f'{seed_id}-eval-{variant}-{case_id}', evaluation['id'])
        try:
            result = mlflow.genai.evaluate(
                data=(native_evaluation_dataset if native_evaluation_dataset is not None else
                      [{'inputs': r['inputs'], 'expectations': r['expectations']} for r in rows]),
                predict_fn=predict_fn, scorers=scorers)
            evaluation.update(status='created', run_id=result.run_id, metrics=plain(result.metrics))
            table = plain(result.result_df.to_dict(orient='records')) if result.result_df is not None else []
            save_json(output_dir / f'evaluation-{variant}.json', table)
            evaluation['results_file'] = f'evaluation-{variant}.json'
        except Exception as exc:
            evaluation.update(status='failed', error_type=type(exc).__name__)
        checkpoint()
    mlflow.flush_trace_async_logging()
    # Store a final manifest and scorer source as source-side MLflow artifacts.
    with mlflow.start_run(run_id=manifest['artifact_run_id']):
        for evaluation in manifest['evaluations']:
            if evaluation.get('results_file'):
                artifact_path = 'seed/evaluations/' + evaluation['results_file']
                mlflow.log_dict(json.loads((output_dir / evaluation['results_file']).read_text()), artifact_path)
                evaluation['results_artifact'] = artifact_path
        mlflow.log_dict(manifest, 'seed/manifest.json')
        mlflow.log_dict({'scorers': manifest['scorers']}, 'seed/scorers.json')
    checkpoint()
    print(f'Seed complete. Manifest: {output_dir / "manifest.json"}')
    print('Optional failures are recorded in resources; inspect before migrating.')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experiment', required=True, help='Databricks absolute experiment path')
    parser.add_argument('--tracking-uri', default='databricks')
    parser.add_argument('--uc-prefix', help='Existing catalog.schema with dataset/prompt permissions')
    args = parser.parse_args()
    seed(args.output, args.experiment, args.tracking_uri, args.uc_prefix)
