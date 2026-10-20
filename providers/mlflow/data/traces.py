"""Discover traces and migrate projects, span hierarchies, and feedback."""
from utils.mlflow import bounded


def export_traces(mlflow, experiment_id, limit=10000):
    from export_mlflow import normalize_trace
    rows = bounded(mlflow.search_traces(locations=[str(experiment_id)],
        return_type='list', max_results=limit), limit, 'traces')
    return [normalize_trace(t, {'evaluation': None, 'discovered': True}) for t in rows]


def migrate_traces(plan, destination, perform, dataset_id):
    for name, item in plan['projects'].items():
        perform('project:' + name, lambda n=name, i=item: destination.project(n, i, dataset_id, plan))
    if hasattr(destination, 'runs'):
        destination.runs(plan['runs'], perform)
    else:
        for item in plan['runs']:
            perform('run:' + item['id'], lambda i=item: destination.run(i))
    for item in plan['feedback']:
        perform('feedback:' + item['feedback_id'], lambda i=item: destination.feedback(i))
