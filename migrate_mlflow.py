"""Plan by default; --apply writes the reviewed plan to LangSmith."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import time
from stack_common import digest, save_json
from migration_core import plan_bundle
from providers.mlflow.data.datasets import migrate_datasets
from providers.mlflow.data.prompts import migrate_prompts
from providers.mlflow.data.traces import migrate_traces


from utils.mlflow_destination import LangSmithDestination, assert_run_matches


def apply_plan(plan, destination, state_path, include_prompts=False):
    state_path = Path(state_path)
    identity = {'bundle_digest': plan['bundle_digest'], 'prefix': plan['prefix'],
                'destination': destination.identity}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['identity'] != identity:
            raise ValueError('State belongs to another bundle, prefix, or destination')
    else:
        state = {'identity': identity, 'completed': {}, 'status': 'running'}
    state['status'] = 'running'
    save_json(state_path, state)
    def perform(key, operation):
        if key in state['completed']:
            return state['completed'][key]
        result = operation()
        state['completed'][key] = result
        save_json(state_path, state)
        return result
    dataset_id = migrate_datasets(plan, destination, perform)
    migrate_traces(plan, destination, perform, dataset_id)
    if include_prompts:
        migrate_prompts(plan, destination, perform)
    for name, item in plan['projects'].items():
        if item['kind'] == 'evaluation':
            project_id = state['completed']['project:' + name]
            perform('finalize:' + name, lambda p=project_id: destination.finalize(p))
    state['status'] = 'applied_requires_validation'
    state['prompts_applied'] = bool(include_prompts or state.get('prompts_applied'))
    save_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--prefix', default='mlflow-demo')
    parser.add_argument('--report', default='migration-plan.json')
    parser.add_argument('--state', default='migration-state.json')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--include-prompts', action='store_true', help='Requires langchain-core; creates version snapshots')
    args = parser.parse_args()
    bundle = json.loads(Path(args.bundle).read_text())
    plan = plan_bundle(bundle, args.prefix)
    save_json(args.report, plan)
    print({k: len(plan[k]) for k in ('examples', 'projects', 'runs', 'feedback', 'prompts', 'skipped')})
    if args.apply:
        apply_plan(plan, LangSmithDestination(), args.state, args.include_prompts)
        print('Applied supported resources. Run validate_migration.py before claiming completion.')
    else:
        print('Dry run only. Review source_inventory, skipped, and limitations in ' + args.report)


if __name__ == '__main__':
    main()
