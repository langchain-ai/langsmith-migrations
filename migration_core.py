"""Pure transformation layer; offline validation and plans require only Python."""
from datetime import datetime, timezone
import re
from uuid import UUID, uuid5
from stack_common import SCHEMA_VERSION, digest

NAMESPACE = UUID('d789cfb3-1a4a-4daf-a9e8-b2ad810188a8')
TYPE_MAP = {'AGENT': 'chain', 'CHAIN': 'chain', 'TOOL': 'tool', 'RETRIEVER': 'retriever',
            'LLM': 'llm', 'CHAT_MODEL': 'llm', 'EMBEDDING': 'embedding',
            'PARSER': 'parser', 'RERANKER': 'chain', 'UNKNOWN': 'chain'}


def stable_id(source, kind, key):
    return str(uuid5(NAMESPACE, f'{source["seed_id"]}/{kind}/{key}'))


def timestamp(ns):
    if not isinstance(ns, int) or ns <= 0:
        raise ValueError('Missing or invalid original nanosecond timestamp')
    seconds, nanos = divmod(ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=nanos // 1000)


def object_io(value):
    return value if isinstance(value, dict) else {'value': value}


def convert_prompt(template):
    if not isinstance(template, str) or '{%' in template:
        raise ValueError('Only simple text prompts with {{identifier}} variables are supported')
    pattern = r'\{\{\s*([A-Za-z_][A-Za-z_0-9]*)\s*\}\}'
    remainder = re.sub(pattern, '', template)
    if '{' in remainder or '}' in remainder:
        raise ValueError('Complex or literal prompt braces require manual conversion')
    return re.sub(pattern, r'{\1}', template)


def ordered_spans(trace):
    spans = trace['spans']
    by_id = {s['id']: s for s in spans}
    if len(by_id) != len(spans):
        raise ValueError('Duplicate span ID')
    roots = [s for s in spans if not s['parent_id']]
    if len(roots) != 1:
        raise ValueError('A complete trace must have exactly one root')
    visiting, seen, ordered = set(), set(), []
    def visit(span):
        if span['id'] in seen:
            return
        if span['id'] in visiting:
            raise ValueError('Span hierarchy contains a cycle')
        visiting.add(span['id'])
        if span['parent_id']:
            if span['parent_id'] not in by_id:
                raise ValueError('Orphan span; refusing to flatten hierarchy')
            visit(by_id[span['parent_id']])
        visiting.remove(span['id'])
        seen.add(span['id'])
        ordered.append(span)
    for span in sorted(spans, key=lambda s: (s['start_ns'], s['id'])):
        visit(span)
    return ordered


def plan_bundle(bundle, prefix='mlflow-demo'):
    if bundle.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported bundle schema')
    if bundle.get('export_errors'):
        raise ValueError('Incomplete source export')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,39}', prefix):
        raise ValueError('Prefix must be lowercase letters/digits/hyphens, max 40 characters')
    source = bundle['source']
    id_source = {**source, 'seed_id': source['seed_id'] + '/' + prefix}
    namespace = prefix + '-' + source['seed_id'][:12]
    dataset = bundle['dataset']
    examples = []
    row_ids = set()
    for row in dataset['records']:
        if row['id'] in row_ids:
            raise ValueError('Duplicate dataset row ID')
        if not isinstance(row['inputs'], dict) or not isinstance(row['expectations'], dict):
            raise ValueError('Dataset inputs and expectations must be dictionaries')
        row_ids.add(row['id'])
        examples.append({'id': stable_id(id_source, 'example', row['id']), 'source_id': row['id'],
                         'inputs': row['inputs'], 'outputs': row['expectations'],
                         'metadata': {**row.get('metadata', {}), 'mlflow_source_row': row['id']}})
    evaluations = {e['id']: e for e in bundle['evaluations']}
    projects = {namespace + '-traces': {'kind': 'tracing', 'metadata': {'mlflow_source': source}}}
    if source.get('discovery') and not bundle['traces']:
        projects = {}
    for e in evaluations.values():
        if e['status'] == 'created':
            projects[namespace + '-eval-' + e['variant']] = {
                'kind': 'evaluation', 'metadata': {'mlflow_source': source,
                'source_evaluation': {k: v for k, v in e.items() if k != 'rows'}}}
    runs, feedback, skipped = [], [], []
    seen_trace_ids, seen_assessments = set(), set()
    for trace in bundle['traces']:
        if trace['id'] in seen_trace_ids:
            raise ValueError('Duplicate source trace ID')
        seen_trace_ids.add(trace['id'])
        spans = ordered_spans(trace)
        root = next(s for s in spans if not s['parent_id'])
        root_id = stable_id(id_source, 'run', trace['id'] + '/' + root['id'])
        ids = {s['id']: stable_id(id_source, 'run', trace['id'] + '/' + s['id']) for s in spans}
        orders = {}
        evaluation_id = trace['link'].get('evaluation')
        e = evaluations.get(evaluation_id)
        if evaluation_id and e is None:
            raise ValueError('Trace references an unknown evaluation')
        project = namespace + ('-eval-' + e['variant'] if e and e['status'] == 'created' else '-traces')
        reference_id = None
        if e and e['status'] == 'created':
            case_id = trace['link']['case_id']
            if case_id not in row_ids:
                raise ValueError('Evaluation trace references a missing example')
            reference_id = stable_id(id_source, 'example', case_id)
        for s in spans:
            start, end = timestamp(s['start_ns']), timestamp(s['end_ns'])
            if end < start:
                raise ValueError('Span ends before it starts')
            segment = start.strftime('%Y%m%dT%H%M%S%fZ') + ids[s['id']]
            orders[s['id']] = ((orders[s['parent_id']] + '.') if s['parent_id'] else '') + segment
            metadata = {'mlflow_trace_id': trace['id'], 'mlflow_span_id': s['id'],
                        'mlflow_span_type': s['type'], 'mlflow_attributes': s['attributes'],
                        'mlflow_trace_metadata': trace['metadata'], 'mlflow_trace_tags': trace['tags'],
                        'mlflow_link': trace['link'], 'source_start_ns': s['start_ns'], 'source_end_ns': s['end_ns']}
            if trace.get('session_id'):
                metadata['thread_id'] = trace['session_id']
            if s['type'] not in TYPE_MAP:
                skipped.append({'kind': 'span_type', 'id': s['id'], 'reason': f"Unknown type {s['type']} mapped to chain"})
            runs.append({'id': ids[s['id']], 'trace_id': root_id,
                         'parent_run_id': ids.get(s['parent_id']), 'name': s['name'],
                         'run_type': TYPE_MAP.get(s['type'], 'chain'), 'inputs': object_io(s['inputs']),
                         'outputs': object_io(s['outputs']) if s['outputs'] is not None else None,
                         'start_time': start.isoformat(), 'end_time': end.isoformat(),
                         'dotted_order': orders[s['id']], 'project_name': project,
                         'error': (s['error'] or 'Source span status ERROR') if s['status'] == 'ERROR' else None,
                         'extra': {'metadata': metadata}, 'tags': (['migrated:mlflow'] if source.get('discovery') else ['migrated:mlflow', 'synthetic:demo']),
                         'reference_example_id': reference_id if s['id'] == root['id'] else None})
        for a in trace.get('assessments', []):
            reason = None
            if a['kind'] != 'feedback': reason = 'Only feedback assessments are mapped; raw assessment preserved in bundle'
            elif a.get('valid') is False or a.get('overrides'): reason = 'Invalidated/override assessment requires manual history mapping'
            elif a.get('error'): reason = 'Scorer error is preserved in bundle, not converted into a score'
            elif isinstance(a.get('value'), list) or a.get('value') is None: reason = 'Unsupported or missing feedback value'
            elif a.get('span_id') and a['span_id'] not in ids: reason = 'Assessment references a missing span'
            elif not a.get('id'): reason = 'Assessment has no stable source ID'
            if reason:
                skipped.append({'kind': 'assessment', 'id': a.get('id'), 'reason': reason})
                continue
            key = trace['id'] + '/' + a['id']
            if key in seen_assessments:
                raise ValueError('Duplicate assessment ID within a trace')
            seen_assessments.add(key)
            value = a['value']
            feedback.append({'feedback_id': stable_id(id_source, 'feedback', key),
                             'run_id': ids[a['span_id']] if a.get('span_id') else root_id,
                             'key': a['name'], 'score': value if isinstance(value, (bool, int, float)) else None,
                             'value': value if isinstance(value, (str, dict)) else None,
                             'comment': a.get('rationale'), 'feedback_source_type': 'api',
                             'source_info': {'mlflow_assessment_id': a['id'], 'mlflow_source': a['source'],
                                             'mlflow_metadata': a.get('metadata'), 'mlflow_raw': a.get('raw')}})
    prompts = []
    for p in bundle['prompts']:
        try:
            converted = convert_prompt(p['template'])
        except ValueError as exc:
            skipped.append({'kind': 'prompt', 'id': p['version'], 'reason': str(exc)})
            continue
        prompt_key = digest([p['name'], p['version']])[:16] if source.get('discovery') else 'support-v' + p['version']
        prompts.append({'identifier': namespace + '-' + prompt_key,
                        'template': converted, 'source': p})
    return {'schema_version': SCHEMA_VERSION, 'source': source, 'bundle_digest': digest(bundle),
            'prefix': prefix, 'dataset_name': namespace + '-dataset', 'examples': examples,
            'create_dataset': bool(examples) or 'native' in dataset or not source.get('discovery'),
            'projects': projects, 'runs': runs, 'feedback': feedback, 'prompts': prompts,
            'scorers': bundle['scorers'], 'skipped': skipped, 'source_inventory': bundle['inventory'],
            'evaluation_trace_reconciliation': bundle.get('evaluation_trace_reconciliation', []),
            'source_evaluations': [{k: v for k, v in e.items() if k != 'rows'} for e in bundle['evaluations']],
            'limitations': [
                ('Experiment-scoped discovery; historical evaluation relationships are archived, not recreated.' if source.get('discovery') else 'Inventory-scoped snapshot; linked datasets and prompts are selected by manifest.'),
                'Prompt versions become separate private prompt snapshots; aliases and native version history are not recreated.',
                ('Scorer names and sampling/filter configuration are archived; scorer code is not discovered or recreated.' if source.get('discovery') else 'Custom scorer source is archived only. The three known demo scorers have hand-written LangSmith adapters.'),
                ('Historical evaluation runs remain source metadata; imported traces do not recreate LangSmith evaluation experiments.' if source.get('discovery') else 'Evaluation projects link imported trace roots to reference examples; source aggregate metrics remain metadata.'),
                'Nanosecond timestamps are truncated to microseconds; original values remain metadata.',
                'Session IDs map to metadata.thread_id; UI conversation rendering is deployment-dependent.',
                'Token/cost fields remain source attributes; no normalized LangSmith cost accounting is claimed.',
                'Assessment authors/timestamps remain provenance; destination feedback is created by the migration API identity.',
                'Binary attachments, model registry, serving, permissions, schedules, and online scorer configuration are not migrated.',
            ]}


def replay_plan(plan, anchor):
    """Replay each trace near a fixed UTC anchor, retaining intra-trace timing."""
    from copy import deepcopy
    from datetime import timedelta
    anchor = datetime.fromisoformat(anchor)
    if anchor.tzinfo is None:
        raise ValueError('Replay anchor must include a timezone')
    anchor = anchor.astimezone(timezone.utc)
    result = deepcopy(plan)
    policy = {'mode': 'replay', 'anchor': anchor.isoformat()}
    result['timestamp_policy'] = policy
    result['bundle_digest'] = digest({'source_bundle': plan['bundle_digest'], 'timestamp_policy': policy})
    groups = {}
    for run in result['runs']:
        groups.setdefault(run['trace_id'], []).append(run)
    for index, (_, runs) in enumerate(sorted(groups.items())):
        latest = max(datetime.fromisoformat(r['end_time']) for r in runs)
        earliest = min(datetime.fromisoformat(r['start_time']) for r in runs)
        offset = anchor - timedelta(seconds=index) - latest
        if latest - earliest + timedelta(seconds=index) >= timedelta(hours=23):
            raise ValueError('Replay exceeds the 23-hour ingestion budget; split this export')
        orders = {}
        for run in runs:
            metadata = run['extra']['metadata']
            metadata.update({'source_start_time': run['start_time'], 'source_end_time': run['end_time'],
                             'migration_timestamp_policy': policy})
            for key in ('start_time', 'end_time'):
                run[key] = (datetime.fromisoformat(run[key]) + offset).isoformat()
            segment = datetime.fromisoformat(run['start_time']).strftime('%Y%m%dT%H%M%S%fZ') + run['id']
            parent = run['parent_run_id']
            orders[run['id']] = (orders[parent] + '.' if parent else '') + segment
            run['dotted_order'] = orders[run['id']]
    result['limitations'].append('Replay dates are ingestion dates; original dates remain metadata. Intra-trace durations are preserved; gaps between traces are not.')
    return result


def prepare_plans(bundles, prefix, output_dir, replay=False):
    """Persist a replay anchor so previews, writes and retries use identical dates."""
    from pathlib import Path
    import json
    from stack_common import save_json
    policy_path = Path(output_dir) / 'timestamp-policy.json'
    identity = {'bundle_digests': [digest(b) for b in bundles], 'prefix': prefix, 'replay': replay}
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        if policy['identity'] != identity:
            raise ValueError('Timestamp policy belongs to another bundle, prefix or mode; use a separate output directory')
    else:
        policy = {'identity': identity, 'anchor': datetime.now(timezone.utc).isoformat() if replay else None}
        save_json(policy_path, policy)
    plans = [plan_bundle(b, prefix=prefix) for b in bundles]
    return [replay_plan(p, policy['anchor']) for p in plans] if replay else plans
