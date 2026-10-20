"""MLflow resource-migration entry point; source setup is deliberately separate."""
import argparse
import os
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_mlflow import export
from migration_core import plan_bundle, prepare_plans
from migrate_mlflow import apply_plan, LangSmithDestination
from validate_migration import validate_live
from utils.mlflow_destination import group_migrated_resources
from stack_common import save_json
from discover_mlflow import export_experiment, list_experiments


def migrate_mlflow(*, output_dir, manifest_path=None, bundle_path=None,
                   tracking_uri=None, prefix='mlflow-migration', apply=False,
                   include_prompts=True, experiment_id=None, replay_timestamps=False):
    """Export existing resources, preview conversion, optionally import and read back.

    Experiment discovery and the optional manifest adapter share the resource modules
    under providers/mlflow/data. Source discovery stays experiment-scoped.
    """
    if sum(x is not None for x in (manifest_path, bundle_path, experiment_id)) != 1:
        raise ValueError('Provide exactly one of experiment_id, manifest_path or bundle_path.')
    output_dir = Path(output_dir)
    if experiment_id is not None:
        bundle = export_experiment(experiment_id, output_dir / 'bundle.json', tracking_uri or 'databricks')
    elif manifest_path is not None:
        bundle = export(manifest_path, output_dir / 'bundle.json',
                        tracking_uri=tracking_uri, discover_traces=True)
    else:
        bundle = json.loads(Path(bundle_path).read_text())
    if bundle.get('format') == 'mlflow-discovery-v1':
        if bundle['export_errors']:
            raise ValueError('Incomplete discovery bundle; inspect coverage report.')
        # Validate every partition before making any destination write.
        prepare_plans(bundle['bundles'], prefix, output_dir, replay_timestamps)
        summaries = []
        for part in bundle['bundles']:
            directory = output_dir / part['source']['seed_id']
            snapshot = directory / 'bundle.json'
            save_json(snapshot, part)
            summaries.append(migrate_mlflow(output_dir=directory, bundle_path=snapshot,
                prefix=prefix, apply=apply, include_prompts=include_prompts,
                replay_timestamps=replay_timestamps))
        summary = {'mode': 'apply' if apply else 'preview', 'coverage': bundle['coverage'],
                   'limitations': bundle['limitations'], 'resources': summaries}
        if apply:
            summary['resource_grouping'] = group_migrated_resources(
                LangSmithDestination().client, output_dir, bundle['bundles'], prefix)
        save_json(output_dir / 'migration-summary.json', summary)
        return summary
    plan = prepare_plans([bundle], prefix, output_dir, replay_timestamps)[0]
    save_json(output_dir / 'migration-plan.json', plan)
    summary = {'mode': 'apply' if apply else 'preview',
               'source': bundle['source'],
               'counts': {k: len(plan[k]) for k in ('examples', 'prompts', 'runs', 'feedback', 'projects')},
               'skipped': plan['skipped'], 'limitations': plan['limitations']}
    if apply:
        destination = LangSmithDestination()
        state = apply_plan(plan, destination, output_dir / 'migration-state.json',
                           include_prompts=include_prompts)
        summary['validation'] = validate_live(plan, destination, state)
        save_json(output_dir / 'validation-report.json', summary['validation'])
    save_json(output_dir / 'migration-summary.json', summary)
    if apply and not summary['validation']['passed']:
        raise RuntimeError('Destination read-back failed; inspect validation-report.json.')
    return summary


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--bundle')
    source.add_argument('--experiment', help='Discover/export by experiment ID; no manifest needed')
    source.add_argument('--list-experiments', action='store_true')
    parser.add_argument('--output', default=os.getenv('MLFLOW_MIGRATION_OUTPUT', 'migration-export'))
    parser.add_argument('--tracking-uri', default=os.getenv('MLFLOW_TRACKING_URI', 'databricks'))
    parser.add_argument('--prefix', default=os.getenv('MLFLOW_MIGRATION_PREFIX', 'mlflow-migration'))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--replay-timestamps', action='store_true', help='Replay near now, preserving original dates in metadata')
    parser.add_argument('--skip-prompts', action='store_true')
    args = parser.parse_args(argv)
    if args.list_experiments:
        print(json.dumps(list_experiments(args.tracking_uri), indent=2))
        raise SystemExit(0)
    print(json.dumps(migrate_mlflow(output_dir=args.output, manifest_path=args.manifest,
        bundle_path=args.bundle, tracking_uri=args.tracking_uri, prefix=args.prefix,
        apply=args.apply, replay_timestamps=args.replay_timestamps, include_prompts=not args.skip_prompts, experiment_id=args.experiment), indent=2))


def interactive_migration():
    from config import (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_ID,
                        MLFLOW_MIGRATION_OUTPUT, MLFLOW_MIGRATION_PREFIX)
    experiment = MLFLOW_EXPERIMENT_ID
    if not experiment:
        print(json.dumps(list_experiments(MLFLOW_TRACKING_URI), indent=2))
        experiment = input('MLflow experiment ID to migrate: ').strip()
    if not experiment:
        print('Migration cancelled.')
        return
    replay = input('Replay historical timestamps near now (originals retained in metadata)? [y/N]: ').strip().lower() in ('y', 'yes')
    result = migrate_mlflow(output_dir=MLFLOW_MIGRATION_OUTPUT, experiment_id=experiment,
        tracking_uri=MLFLOW_TRACKING_URI, prefix=MLFLOW_MIGRATION_PREFIX, replay_timestamps=replay)
    print(json.dumps(result, indent=2))
    if input('Import this preview into the configured LangSmith workspace? [y/N]: ').strip().lower() not in ('y', 'yes'):
        return
    return migrate_mlflow(output_dir=MLFLOW_MIGRATION_OUTPUT,
        bundle_path=Path(MLFLOW_MIGRATION_OUTPUT) / 'bundle.json',
        prefix=MLFLOW_MIGRATION_PREFIX, apply=True, replay_timestamps=replay)


if __name__ == '__main__':
    cli()
