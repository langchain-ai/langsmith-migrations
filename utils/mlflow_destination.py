"""LangSmith destination access and read-back comparison helpers."""
from datetime import datetime, timezone
import os
import time
from stack_common import digest


def iso(value):
    if value is None:
        return None
    return value.isoformat() if hasattr(value, 'isoformat') else datetime.fromisoformat(value).isoformat()


def assert_run_matches(actual, expected):
    for key in ('name', 'run_type', 'inputs', 'outputs', 'error', 'dotted_order'):
        if getattr(actual, key, None) != expected[key]:
            raise ValueError(f'Destination run mismatch: {expected["id"]} field {key}')
    for key in ('parent_run_id', 'trace_id', 'reference_example_id'):
        value = getattr(actual, key, None)
        if (str(value) if value is not None else None) != expected[key]:
            raise ValueError(f'Destination run mismatch: {expected["id"]} field {key}')
    for key in ('start_time', 'end_time'):
        if iso(getattr(actual, key, None)) != iso(expected[key]):
            raise ValueError(f'Destination run mismatch: {expected["id"]} field {key}')
    metadata = (getattr(actual, 'extra', None) or {}).get('metadata', {})
    for key, value in expected['extra']['metadata'].items():
        if metadata.get(key) != value:
            raise ValueError(f'Destination run metadata mismatch: {key}')


class LangSmithDestination:
    def __init__(self):
        from langsmith import Client
        from langsmith.utils import LangSmithNotFoundError
        self.client = Client(auto_batch_tracing=False)
        self.not_found = LangSmithNotFoundError
        # Workspace identity must be explicit to prevent accidental cross-workspace resume.
        workspace = os.getenv('LANGSMITH_WORKSPACE_ID')
        if not workspace:
            raise ValueError('Set LANGSMITH_WORKSPACE_ID explicitly for migration checkpoints.')
        self.identity = digest({'endpoint': self.client.api_url, 'workspace': workspace})

    def optional(self, function, **kwargs):
        try:
            return function(**kwargs)
        except self.not_found:
            return None

    def dataset(self, plan):
        name = plan['dataset_name']
        obj = self.optional(self.client.read_dataset, dataset_name=name)
        provenance = {'migration_bundle_digest': plan['bundle_digest']}
        if obj is None:
            obj = self.client.create_dataset(dataset_name=name, metadata=provenance,
                description='Imported MLflow dataset; expectations are references.')
        metadata = obj.metadata or {}
        if metadata.get('migration_bundle_digest') != plan['bundle_digest']:
            raise ValueError('Existing dataset does not belong to this exact bundle')
        return str(obj.id)

    def example(self, item, dataset_id):
        obj = self.optional(self.client.read_example, example_id=item['id'])
        if obj is None:
            obj = self.client.create_example(example_id=item['id'], dataset_id=dataset_id,
                inputs=item['inputs'], outputs=item['outputs'], metadata=item['metadata'])
        if (str(obj.dataset_id) != dataset_id or obj.inputs != item['inputs'] or obj.outputs != item['outputs']
                or any((obj.metadata or {}).get(k) != v for k, v in item['metadata'].items())):
            raise ValueError('Existing example differs from the plan')
        return str(obj.id)

    def project(self, name, item, dataset_id, plan):
        obj = self.optional(self.client.read_project, project_name=name)
        metadata = {**item['metadata'], 'migration_bundle_digest': plan['bundle_digest']}
        reference = dataset_id if item['kind'] == 'evaluation' else None
        if obj is None:
            obj = self.client.create_project(project_name=name, metadata=metadata,
                reference_dataset_id=reference, description='Imported MLflow snapshot')
        if (obj.extra or {}).get('metadata', {}).get('migration_bundle_digest') != plan['bundle_digest']:
            raise ValueError('Existing project belongs to a different bundle')
        actual_reference = getattr(obj, 'reference_dataset_id', None)
        if (str(actual_reference) if actual_reference else None) != reference:
            raise ValueError('Project reference dataset mismatch')
        return str(obj.id)

    def finalize(self, project_id):
        obj = self.client.read_project(project_id=project_id)
        if obj.end_time is None:
            self.client.update_project(project_id=project_id, end_time=datetime.now(timezone.utc),
                                       name=obj.name, description=obj.description, project_extra=obj.extra)
        return project_id

    def runs(self, items, perform):
        """Read/check runs in pages; avoid one GET request per span."""
        for project_name in sorted({r['project_name'] for r in items}):
            project = self.client.read_project(project_name=project_name)
            rows = [r for r in items if r['project_name'] == project_name]
            existing = {str(r.id): r for r in self.client.list_runs(project_name=project_name)}
            for start in range(0, len(rows), 50):
                batch = rows[start:start + 50]
                def upload(batch=batch):
                    missing = []
                    for item in batch:
                        if item['id'] in existing:
                            assert_run_matches(existing[item['id']], item)
                        else:
                            payload = dict(item)
                            payload['session_id'] = str(project.id)
                            missing.append(payload)
                    if missing:
                        self.client.batch_ingest_runs(create=missing)
                    return [r['id'] for r in batch]
                perform('run_batch:' + project_name + ':' + str(start), upload)

    def run(self, item):
        obj = self.optional(self.client.read_run, run_id=item['id'])
        if obj is None:
            payload = dict(item)
            for key in ('start_time', 'end_time'):
                payload[key] = datetime.fromisoformat(payload[key])
            self.client.create_run(**payload)
            # Successful HTTP ingestion can precede read visibility. Check before committing state.
            for attempt in range(6):
                obj = self.optional(self.client.read_run, run_id=item['id'])
                if obj is not None:
                    break
                time.sleep(1)
            if obj is None:
                raise RuntimeError('Run not yet readable; rerun migration with the same state file.')
        assert_run_matches(obj, item)
        project = self.client.read_project(project_name=item['project_name'])
        if str(obj.session_id) != str(project.id):
            raise ValueError('Run belongs to a different destination project')
        return item['id']

    def feedback(self, item):
        obj = self.optional(self.client.read_feedback, feedback_id=item['feedback_id'])
        if obj is None:
            obj = self.client.create_feedback(**item)
        for key in ('key', 'score', 'value', 'comment'):
            if getattr(obj, key, None) != item[key]:
                raise ValueError('Existing feedback differs from the plan: ' + key)
        if str(obj.run_id) != item['run_id']:
            raise ValueError('Feedback belongs to a different run')
        return str(obj.id)

    def prompt(self, item):
        from langchain_core.prompts import PromptTemplate
        obj = self.optional(self.client.pull_prompt, prompt_identifier=item['identifier'])
        if obj is None:
            url = self.client.push_prompt(item['identifier'],
                object=PromptTemplate.from_template(item['template']), is_public=False,
                description='MLflow prompt snapshot; source version ' + item['source']['version'])
        else:
            url = None
        obj = self.client.pull_prompt(item['identifier'])
        if getattr(obj, 'template', None) != item['template']:
            raise ValueError('Destination prompt content differs; refusing to overwrite')
        return {'identifier': item['identifier'], 'url': url,
                'source_version': item['source']['version'],
                'destination_metadata': {k: v for k, v in (getattr(obj, 'metadata', None) or {}).items()
                                         if k in ('lc_hub_commit_hash', 'lc_hub_repo', 'lc_hub_owner')}}




def group_migrated_resources(client, output_dir, bundles, prefix):
    """Attach migrated datasets to the agent represented by the tracing project."""
    import json
    from pathlib import Path
    from stack_common import save_json
    base = '/workspaces/current'
    value = prefix + '-' + bundles[0]['source']['seed_id'][:12] + '-traces'
    resources = []
    for bundle in bundles:
        state_path = Path(output_dir) / bundle['source']['seed_id'] / 'migration-state.json'
        state = json.loads(state_path.read_text())
        for key, resource_id in state['completed'].items():
            if key == 'dataset':
                resources.append({'resource_type': 'dataset',
                                  'resource_id': resource_id})
            elif key.startswith('prompt:'):
                prompt = client.get_prompt(resource_id['identifier'])
                if prompt is None:
                    raise ValueError('Imported prompt is missing')
                resources.append({'resource_type': 'prompt', 'resource_id': prompt.id})
    report = attach_agent_resources(client, value, resources)
    save_json(Path(output_dir) / 'resource-grouping.json', report)
    return report


def attach_agent_resources(client, value, resources):
    """Attach resources to the Agent grouping, not a generic Application tag."""
    base = '/workspaces/current'
    if not resources:
        return {'status': 'not_needed', 'resources': []}
    def request(method, path, **kwargs):
        return client.request_with_retries(method, base + path, **kwargs).json()
    keys = request('GET', '/tag-keys')
    key = next((k for k in keys if k['key'] == 'Agent'), None)
    if key is None:
        key = request('POST', '/tag-keys', json={'key': 'Agent'})
    path = '/tag-keys/' + key['id'] + '/tag-values'
    values = request('GET', path)
    tag = next((v for v in values if v['value'] == value), None)
    if tag is None:
        tag = request('POST', path, json={'value': value})
    writes = 0
    for resource in resources:
        def tagged():
            tags = request('GET', '/tags/resource', params=resource)
            return any(v['id'] == tag['id'] for k in tags for v in k.get('values', []))
        if not tagged():
            request('POST', '/taggings', json={**resource, 'tag_value_id': tag['id']})
            writes += 1
        if not tagged():
            raise RuntimeError('Resource grouping tag failed read-back')
    report = {'status': 'verified', 'key': 'Agent', 'value': value,
              'taggings_created': writes, 'resources': resources}
    return report
