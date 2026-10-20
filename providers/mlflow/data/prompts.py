"""Discover prompt versions and migrate them as private snapshots."""
from stack_common import plain
from utils.mlflow import bounded


def export_prompts(mlflow, client, experiment_id, names=(), limit=10000):
    import json
    if getattr(mlflow, 'get_registry_uri', lambda: '')().startswith('databricks-uc'):
        # Managed UC rejects experiment filters. Follow recorded run/trace links
        # instead of exporting unrelated prompts from the surrounding schema.
        found_names = set(names)
        runs = bounded(client.search_runs([str(experiment_id)], max_results=limit), limit, 'prompt-linked runs')
        for run in runs:
            for item in json.loads(run.data.tags.get('mlflow.linkedPrompts', '[]')):
                found_names.add(item['name'])
        for trace in bounded(mlflow.search_traces(locations=[str(experiment_id)],
                return_type='list', max_results=limit), limit, 'prompt-linked traces'):
            for item in json.loads((trace.info.tags or {}).get('mlflow.linkedPrompts', '[]')):
                found_names.add(item['name'])
    else:
        found = bounded(mlflow.genai.search_prompts(
            filter_string=f"experiment_id = '{experiment_id}'", max_results=limit), limit, 'prompts')
        found_names = {p.name for p in found} | set(names)
    result = []
    for name in sorted(found_names):
        escaped = name.replace("'", "''")
        if hasattr(client, 'search_prompt_versions'):
            versions, token = [], None
            while True:
                page = client.search_prompt_versions(name, max_results=min(100, limit), page_token=token)
                versions.extend(page)
                bounded(versions, limit, 'prompt versions')
                token = getattr(page, 'token', None)
                if not token:
                    break
        else:
            versions = bounded(client.search_model_versions(
                filter_string=f"name = '{escaped}'", max_results=limit), limit, 'prompt versions')
        if not versions:
            raise ValueError(f'No readable versions for linked prompt {name}')
        for version in versions:
            pv = client.load_prompt(f'prompts:/{name}/{version.version}')
            result.append({'name': name, 'version': str(version.version),
                'template': plain(pv.template), 'aliases': list(getattr(version, 'aliases', []) or [])})
    return result


def migrate_prompts(plan, destination, perform):
    for item in plan['prompts']:
        perform('prompt:' + item['identifier'], lambda i=item: destination.prompt(i))
