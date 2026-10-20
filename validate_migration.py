"""Validate a bundle offline, or read back migrated resources with --live."""
from pathlib import Path
import argparse
import json
from migration_core import plan_bundle
from stack_common import save_json, digest
from migrate_mlflow import LangSmithDestination, assert_run_matches


def validate_live(plan, destination, state):
    if state['identity'] != {'bundle_digest': plan['bundle_digest'], 'prefix': plan['prefix'], 'destination': destination.identity}:
        raise ValueError('Checkpoint identity does not match this validation target')
    client = destination.client
    checked = {'examples': 0, 'projects': 0, 'runs': 0, 'feedback': 0, 'prompts': 0}
    errors = []
    def check(kind, key, operation):
        try:
            operation()
            checked[kind] += 1
        except Exception as exc:
            errors.append({'kind': kind, 'id': key, 'error_type': type(exc).__name__})
    ds = None
    if plan.get('create_dataset', True):
        ds = client.read_dataset(dataset_name=plan['dataset_name'])
        if (ds.metadata or {}).get('migration_bundle_digest') != plan['bundle_digest']:
            errors.append({'kind': 'dataset', 'id': str(ds.id), 'error_type': 'DatasetProvenanceMismatch'})
    def verify_example(item):
        obj = client.read_example(example_id=item['id'])
        assert str(obj.dataset_id) == str(ds.id)
        assert obj.inputs == item['inputs'] and obj.outputs == item['outputs']
        assert all((obj.metadata or {}).get(k) == v for k, v in item['metadata'].items())
    if ds is not None:
        actual_ids = {str(e.id) for e in client.list_examples(dataset_id=ds.id)}
        if actual_ids != {e['id'] for e in plan['examples']}:
            errors.append({'kind': 'dataset', 'id': str(ds.id), 'error_type': 'ExampleSetMismatch'})
    projects = {}
    actual_runs = {}
    for name, item in plan['projects'].items():
        def verify_project(n=name, i=item):
            obj = client.read_project(project_name=n)
            projects[n] = obj
            assert (obj.extra or {}).get('metadata', {}).get('migration_bundle_digest') == plan['bundle_digest']
            expected_reference = str(ds.id) if i['kind'] == 'evaluation' else None
            assert (str(obj.reference_dataset_id) if obj.reference_dataset_id else None) == expected_reference
            if i['kind'] == 'evaluation':
                assert obj.end_time is not None
            rows = list(client.list_runs(project_name=n))
            actual_runs.update({str(r.id): r for r in rows})
            ids = {str(r.id) for r in rows}
            assert ids == {r['id'] for r in plan['runs'] if r['project_name'] == n}
        check('projects', name, verify_project)
    for item in plan['examples']:
        check('examples', item['id'], lambda i=item: verify_example(i))
    for item in plan['runs']:
        def verify_run(i=item):
            obj = actual_runs[i['id']]
            assert_run_matches(obj, i)
            assert str(obj.session_id) == str(projects[i['project_name']].id)
        check('runs', item['id'], verify_run)
    actual_feedback = {}
    feedback_runs = sorted({i['run_id'] for i in plan['feedback']})
    for start in range(0, len(feedback_runs), 50):
        for obj in client.list_feedback(run_ids=feedback_runs[start:start + 50]):
            actual_feedback[str(obj.id)] = obj
    for item in plan['feedback']:
        def verify_feedback(i=item):
            obj = actual_feedback[i['feedback_id']]
            assert str(obj.run_id) == i['run_id']
            assert all(getattr(obj, k) == i[k] for k in ('key', 'score', 'value', 'comment'))
        check('feedback', item['feedback_id'], verify_feedback)
    if state.get('prompts_applied'):
        for item in plan['prompts']:
            def verify_prompt(i=item):
                obj = client.pull_prompt(i['identifier'])
                assert obj.template == i['template']
            check('prompts', item['identifier'], verify_prompt)
    return {'mode': 'live_readback', 'passed': not errors, 'checked': checked,
            'errors': errors, 'skipped': plan['skipped'], 'limitations': plan['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--prefix', default='mlflow-demo')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--state', default='migration-state.json')
    parser.add_argument('--report', default='validation-report.json')
    args = parser.parse_args()
    bundle = json.loads(Path(args.bundle).read_text())
    plan = plan_bundle(bundle, args.prefix)
    if args.live:
        state = json.loads(Path(args.state).read_text())
        result = validate_live(plan, LangSmithDestination(), state)
    else:
        result = {'mode': 'offline_structure_only', 'passed': True,
                  'bundle_digest': digest(bundle), 'trace_count': len(bundle['traces']),
                  'span_count': len(plan['runs']), 'destination_verified': False,
                  'skipped': plan['skipped'], 'limitations': plan['limitations']}
    save_json(args.report, result)
    print(f'{result["mode"]}: {"PASS" if result["passed"] else "FAIL"}; {args.report}')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
