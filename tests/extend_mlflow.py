"""Populate native demo resources in an existing experiment; no destination writes."""
import json
import os
import re
from pathlib import Path

from seed_mlflow import build_agent
from stack_common import save_json, is_databricks_uri, databricks_registry_uri

SETUP_VERSION = '2026-10-17.7'


def configure_monitoring_warehouse(mlflow, experiment_id, warehouse_id):
    """Background monitoring reads the experiment tag, not the driver's environment."""
    helper = getattr(mlflow.tracing, 'set_databricks_monitoring_sql_warehouse_id', None)
    if helper:
        helper(sql_warehouse_id=warehouse_id, experiment_id=experiment_id)
    else:
        mlflow.MlflowClient().set_experiment_tag(experiment_id,
            'mlflow.monitoring.sqlWarehouseId', warehouse_id)
    if mlflow.get_experiment(experiment_id).tags.get('mlflow.monitoring.sqlWarehouseId') != warehouse_id:
        raise RuntimeError('Monitoring SQL warehouse did not persist on experiment')


def trace_has_model(trace, model_id):
    return trace is not None and (trace.info.trace_metadata or {}).get('mlflow.modelId') == model_id


def review_input(app):
    """Resolve the input class from the installed SDK instead of a versioned module path."""
    import inspect
    import typing
    method = inspect.unwrap(app.create_label_schema)
    module = inspect.getmodule(method)
    cls = getattr(module, 'InputCategorical', None)
    if cls is None:
        annotation = typing.get_type_hints(method).get('input')
        candidates = typing.get_args(annotation) or (annotation,)
        cls = next((t for t in candidates if getattr(t, '__name__', '') == 'InputCategorical'), None)
    if cls is None:
        raise RuntimeError('Installed review SDK does not expose a categorical input type. Create a feedback question in the Reviews UI, then rerun.')
    return cls(options=['yes', 'no'])


def ensure_agent_records(mlflow, manifest, checkpoint):
    """Persist both model IDs before any eventually consistent trace checks."""
    models = manifest.setdefault('agent_versions', {})
    for variant in ('baseline', 'improved'):
        if variant not in models:
            model = mlflow.create_external_model(
                name=f"support-{variant}-{manifest['seed_id'][:8]}", model_type='agent',
                experiment_id=manifest['experiment_id'],
                tags={'synthetic': 'true', 'variant': variant},
                params={'implementation': 'scripted Python tools; no LLM'})
            models[variant] = {'model_id': model.model_id}
            checkpoint()
    return models


def ensure_review_schema(app, schema_input):
    name = 'synthetic_support_quality'
    if name not in {schema.name for schema in app.label_schemas}:
        app.create_label_schema(name=name, type='feedback', title='Is this response acceptable?',
            input=schema_input, instruction='Review this synthetic support interaction.',
            enable_comment=True)
    return name


def configure_warehouse(tracking_uri, workspace=None):
    """Use configured warehouse or select an existing one; never create compute."""
    key = 'MLFLOW_TRACING_SQL_WAREHOUSE_ID'
    if os.environ.get(key):
        return os.environ[key]
    if workspace is None:
        from databricks.sdk import WorkspaceClient
        from mlflow.utils.uri import get_db_info_from_uri
        profile, _ = get_db_info_from_uri(tracking_uri)
        workspace = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    warehouses = [w for w in workspace.warehouses.list() if w.id]
    if not warehouses:
        raise RuntimeError('No SQL warehouse is visible. In Databricks SQL Warehouses, create one or obtain CAN USE access, then retry.')
    warehouses.sort(key=lambda w: (
        getattr(w.state, 'value', w.state) != 'RUNNING',
        not getattr(w, 'enable_serverless_compute', False), w.id))
    chosen = warehouses[0]
    os.environ[key] = chosen.id
    print('UC trace SQL warehouse:', chosen.name, chosen.id)
    return chosen.id


def setup(manifest_path, uc_prefix=None, tracking_uri=None, spark=None, online_scorer=None):
    """Keep the original source; seed a UC-backed replacement before any tracing."""
    import mlflow
    from mlflow.entities.trace_location import UnityCatalog
    from seed_mlflow import seed
    original = Path(manifest_path)
    source = json.loads(original.read_text())
    uri = tracking_uri or source['tracking_uri']
    mlflow.set_tracking_uri(uri)
    if not is_databricks_uri(uri):
        return extend(original, uc_prefix, uri, spark, online_scorer)
    configure_warehouse(uri)
    existing = mlflow.get_experiment(source['experiment_id'])
    if existing.trace_location is not None:
        return extend(original, uc_prefix, uri, spark, online_scorer)
    # A populated experiment cannot acquire a UC trace destination on this backend.
    output = original.parent / 'uc-stack'
    target = output / 'manifest.json'
    ready = output / 'seed-complete.json'
    if ready.exists():
        print('Resuming UC-backed stack:', target)
        return extend(target, uc_prefix, uri, spark, online_scorer)
    if target.exists():
        raise RuntimeError('UC seed was interrupted. Inspect uc-stack/manifest.json before retrying; no resources were deleted.')
    prefix = uc_prefix or source.get('extension', {}).get('uc_prefix')
    if not prefix and source['dataset'].get('native'):
        prefix = source['dataset']['native']['name'].rsplit('.', 1)[0]
    if not prefix:
        raise ValueError('No storage namespace recorded; supply UC_PREFIX for the fresh UC stack.')
    validate_namespace(prefix)
    catalog, schema = prefix.split('.')
    name = source['experiment_name'] + '-uc-stack'
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name,
            trace_location=UnityCatalog(catalog_name=catalog, schema_name=schema))
        experiment = mlflow.get_experiment(experiment_id)
    if experiment.trace_location is None:
        raise RuntimeError('Replacement experiment has no UC destination; no traces were generated.')
    if mlflow.search_traces(locations=[experiment.experiment_id], max_results=1, return_type='list'):
        raise RuntimeError('Replacement experiment already contains traces without a completed seed inventory; inspect before retrying.')
    print('Creating UC-backed source stack:', name)
    configure_monitoring_warehouse(mlflow, experiment.experiment_id,
                                  os.environ['MLFLOW_TRACING_SQL_WAREHOUSE_ID'])
    seed(output, name, uri, prefix)
    save_json(ready, {'experiment_id': experiment.experiment_id, 'original_manifest': str(original)})
    print('Migration inventory:', target)
    return extend(target, prefix, uri, spark, online_scorer)


def response_present(outputs):
    """Online scorer: uses observed output, without offline expectations or an LLM."""
    return (isinstance(outputs, dict) and isinstance(outputs.get('answer'), str)
            and bool(outputs['answer'].strip()))


def successful_monitor_assessments(trace, name):
    return [a for a in trace.info.assessments or []
            if a.name == name and a.feedback is not None
            and not a.feedback.error and a.valid is not False]


def validate_namespace(value):
    if value is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError('UC_PREFIX must be an existing catalog.schema (simple identifiers).')


def discover_namespaces(spark):
    """Prefer the current namespace, then default schemas; never change SQL context."""
    current = tuple(spark.sql('SELECT current_catalog(), current_schema()').first())
    candidates = []
    for row in spark.sql('SHOW CATALOGS').collect():
        catalog = row[0]
        if catalog.lower() in {'system', 'hive_metastore', 'samples'}:
            continue
        quoted = '`' + catalog.replace('`', '``') + '`'
        try:
            schemas = spark.sql(f'SHOW SCHEMAS IN {quoted}').collect()
        except Exception:
            continue
        for schema_row in schemas:
            schema = schema_row[0]
            if schema.lower() == 'information_schema':
                continue
            name = f'{catalog}.{schema}'
            try:
                validate_namespace(name)
            except ValueError:
                continue
            candidates.append((catalog, schema))
    candidates = sorted(set(candidates), key=lambda x: (
        x != current, x[0] != current[0], x[1] != 'default', x))
    return ['.'.join(pair) for pair in candidates]


def extend(manifest_path, uc_prefix=None, tracking_uri=None, spark=None, online_scorer=None):
    """Resume from a local manifest. Single writer; checkpoint each successful write.

    Interrupted writes before a checkpoint may require manual duplicate reconciliation.
    Backend failures are reported per resource; they never imply tab completion.
    """
    validate_namespace(uc_prefix)
    import mlflow
    from mlflow import MlflowClient
    from mlflow.genai.scorers import scorer, list_scorers
    from stack_common import SCORERS

    path = Path(manifest_path)
    m = json.loads(path.read_text())
    m['setup_version'] = SETUP_VERSION
    print('MLflow setup version:', SETUP_VERSION)
    uri = tracking_uri or m['tracking_uri']
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_id=m['experiment_id'])
    client = MlflowClient()
    db = is_databricks_uri(uri)
    if db:
        mlflow.set_registry_uri(databricks_registry_uri(uri))
    report = m['tab_report'] = {}
    state = m.setdefault('extension', {})
    if db and not uc_prefix:
        native = m['dataset'].get('native')
        uc_prefix = native['name'].rsplit('.', 1)[0] if native else state.get('uc_prefix')
    candidates = [uc_prefix] if uc_prefix else []
    if db and not candidates and spark is not None:
        try:
            candidates = discover_namespaces(spark)
            print('Discovered storage candidates:', candidates)
        except Exception as exc:
            state['namespace_discovery_error'] = type(exc).__name__

    def checkpoint():
        save_json(path, m)

    def attempt(tab, fn):
        try:
            report[tab] = {'status': 'verified', **fn()}
        except Exception as exc:
            report[tab] = {'status': 'failed', 'error_type': type(exc).__name__}
            if type(exc) is RuntimeError:
                report[tab]['message'] = str(exc)[:1500]
            response = getattr(exc, 'response', None)
            status = getattr(response, 'status_code', None)
            if isinstance(status, int):
                report[tab]['http_status'] = status
            if tab == 'Reviews':
                report[tab]['operation'] = state.get('review_operation')
            # Keep structured API messages, never request headers or full HTTP objects.
            payload = getattr(exc, 'json', None)
            if not isinstance(payload, dict) and response is not None:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
            if isinstance(payload, dict):
                report[tab]['error_code'] = payload.get('error_code')
                message = str(payload.get('message', ''))
                message = re.sub(r'https?://\S+', '[URL]', message)
                message = re.sub(r'(?i)(bearer\s+|token[=:]\s*)\S+', r'\1[REDACTED]', message)
                report[tab]['message'] = message[:1500]
        checkpoint()

    def dataset():
        nonlocal uc_prefix
        native = m['dataset'].get('native')
        if native:
            ds = mlflow.genai.datasets.get_dataset(name=native['name'])
        else:
            failures = state['namespace_attempts'] = []
            for prefix in (candidates if db else [uc_prefix]):
                name = (prefix + '.' if prefix else '') + 'support_cases_' + m['seed_id'][:8]
                try:
                    ds = mlflow.genai.datasets.create_dataset(name=name, experiment_id=m['experiment_id'])
                except Exception as exc:
                    code = str(getattr(exc, 'error_code', ''))
                    failures.append({'namespace': prefix, 'error_type': type(exc).__name__, 'code': code})
                    checkpoint()
                    # Only try another schema for a definite permission rejection.
                    # Other failures can be ambiguous remote writes; do not multiply resources.
                    if code in {'PERMISSION_DENIED', 'INSUFFICIENT_PERMISSIONS'}:
                        continue
                    raise
                uc_prefix = prefix
                state['uc_prefix'] = prefix
                print('Selected dataset/prompt location:', prefix)
                break
            else:
                raise PermissionError('No discovered namespace permitted dataset creation')
            m['dataset']['native'] = {'name': ds.name, 'id': ds.dataset_id}
            checkpoint()
        ds.merge_records([{'inputs': r['inputs'], 'expectations': r['expectations']}
                          for r in m['dataset']['records']])
        ds = mlflow.genai.datasets.get_dataset(name=ds.name)
        if len(ds.to_df()) < len(m['dataset']['records']):
            raise RuntimeError('Native dataset read-back is missing records')
        m['resources']['native_dataset'] = {'status': 'created', 'result': m['dataset']['native']}
        return {'name': ds.name, 'records_submitted': len(m['dataset']['records'])}

    def prompts():
        for p in m['prompts']:
            if not p.get('native'):
                name = (uc_prefix + '.' if uc_prefix else '') + 'support_prompt_' + m['seed_id'][:8]
                pv = mlflow.genai.register_prompt(name=name, template=p['template'],
                                                  commit_message='Synthetic support demo version ' + p['version'])
                p['native'] = {'name': pv.name, 'version': str(pv.version)}
                checkpoint()
            n = p['native']
            for alias in p['aliases']:
                mlflow.genai.set_prompt_alias(n['name'], alias, int(n['version']))
            # Loading during a run links the prompt to that run/experiment.
            with mlflow.start_run(run_id=m['artifact_run_id']):
                mlflow.genai.load_prompt(f"prompts:/{n['name']}/{n['version']}")
        m['resources']['native_prompts'] = {'status': 'created'}
        return {'versions': [p['native'] for p in m['prompts']]}

    if db and not candidates and not m['dataset'].get('native'):
        report['Datasets'] = {'status': 'blocked', 'reason': 'No eligible namespace discovered. Supply UC_PREFIX or check catalog access.'}
    else:
        attempt('Datasets', dataset)
    if db and not uc_prefix and not all(p.get('native') for p in m['prompts']):
        report['Prompts'] = {'status': 'blocked', 'reason': 'Set UC_PREFIX to a writable catalog.schema.'}
    else:
        attempt('Prompts', prompts)

    def storage():
        from mlflow.entities.trace_location import UnityCatalog
        existing = mlflow.get_experiment(m['experiment_id']).trace_location
        location = existing
        if location is None:
            if not uc_prefix:
                raise ValueError('No writable UC namespace selected')
            catalog, schema = uc_prefix.split('.')
            location = UnityCatalog(catalog_name=catalog, schema_name=schema,
                                    table_prefix=m['experiment_id'])
        experiment = mlflow.set_experiment(experiment_id=m['experiment_id'], trace_location=location)
        if experiment.trace_location is None:
            raise RuntimeError('Trace location was not returned')
        state['trace_storage'] = str(experiment.trace_location)
        return {'location': state['trace_storage'], 'historical_traces': 'not backfilled'}
    if db:
        attempt('UC trace storage', storage)

    def scorers():
        existing = {s.name for s in list_scorers(experiment_id=m['experiment_id'])}
        for name, fn in SCORERS.items():
            if name not in existing:
                scorer(name=name)(fn).register(experiment_id=m['experiment_id'])
        names = {s.name for s in list_scorers(experiment_id=m['experiment_id'])}
        if not set(SCORERS).issubset(names):
            raise RuntimeError('Registered scorers missing on read-back')
        return {'names': sorted(names)}
    attempt('Scorers', scorers)

    def monitoring():
        from mlflow.genai.scorers import ScorerSamplingConfig
        warehouse_id = configure_warehouse(uri)
        configure_monitoring_warehouse(mlflow, m['experiment_id'], warehouse_id)
        name = 'synthetic_response_present'
        registered = {s.name: s for s in list_scorers(experiment_id=m['experiment_id'])}
        monitor = registered.get(name)
        if monitor is None and online_scorer is not None:
            monitor = online_scorer.register(experiment_id=m['experiment_id'])
            checkpoint()
        elif monitor is None:
            raise RuntimeError('Define the online scorer in the Databricks notebook and pass online_scorer to setup.')
        sampling = ScorerSamplingConfig(sample_rate=1.0,
            filter_string=f"tags.`monitoring_batch` = '{m['seed_id']}'")
        if monitor.sample_rate != 1.0 or monitor.filter_string != sampling.filter_string:
            monitor.start(experiment_id=m['experiment_id'], sampling_config=sampling)
        persisted = {s.name: s for s in list_scorers(experiment_id=m['experiment_id'])}[name]
        if persisted.sample_rate != 1.0 or persisted.filter_string != sampling.filter_string:
            raise RuntimeError('Online scorer configuration did not persist')
        state['monitor_name'] = name
        return {'status': 'pending', 'reason': 'Configured; waiting for automatic assessments.'}
    if db:
        attempt('Online monitoring', monitoring)

    def models():
        models = ensure_agent_records(mlflow, m, checkpoint)
        agent = build_agent(mlflow)
        row = m['dataset']['records'][0]
        pending = []
        for idx, variant in enumerate(('baseline', 'improved')):
            model_id = models[variant]['model_id']
            client.get_logged_model(model_id)
            # One additional tool-bearing trace per version; no evaluation rerun.
            # Repeat only if native prompts were added since the previous attempt.
            prompt_key = m['prompts'][idx].get('native')
            if not models[variant].get('trace_id') or models[variant].get('prompt') != prompt_key:
                session_id = f"{m['seed_id']}-version-{variant}"
                with mlflow.set_active_model(model_id=model_id):
                    with mlflow.start_span(name='versioned_support', span_type='AGENT') as root:
                        trace_id = root.trace_id
                        mlflow.update_current_trace(model_id=model_id, session_id=session_id)
                        root.set_inputs(row['inputs'])
                        root.set_outputs(agent(row['id'], row['inputs']['messages'], variant,
                                               session_id, m['prompts'][idx]))
                if not trace_id:
                    raise RuntimeError('No model-linked trace returned')
                models[variant].update(trace_id=trace_id, prompt=prompt_key)
                m['traces'].append({'trace_id': trace_id, 'case_id': row['id'],
                                    'variant': variant, 'session_id': session_id, 'evaluation': None})
                checkpoint()
            mlflow.flush_trace_async_logging()
            persisted_trace = mlflow.get_trace(models[variant]['trace_id'])
            if not trace_has_model(persisted_trace, model_id):
                pending.append(variant)
        if pending:
            return {'status': 'pending', 'versions': models,
                    'reason': 'Model records exist; trace links not yet visible for: ' + ', '.join(pending)}
        return {'versions': models, 'deployment': 'external code; no serving endpoint'}
    attempt('Agent versions', models)

    def native_evaluations():
        from stack_common import plain
        if any(not m.get('agent_versions', {}).get(v, {}).get('model_id') for v in ('baseline', 'improved')):
            return {'status': 'blocked', 'reason': 'Both agent model IDs must be created first.'}
        if report['Datasets']['status'] != 'verified' or report['Prompts']['status'] != 'verified':
            raise RuntimeError('Native dataset and prompts must succeed first')
        ds = mlflow.genai.datasets.get_dataset(name=m['dataset']['native']['name'])
        agent = build_agent(mlflow)
        completed = state.setdefault('native_evaluations', {})
        for idx, variant in enumerate(('baseline', 'improved')):
            if variant in completed:
                client.get_run(completed[variant]['run_id'])
                continue
            def predict_fn(case_id, messages):
                return agent(case_id, messages, variant,
                             f"{m['seed_id']}-native-{variant}-{case_id}", m['prompts'][idx])
            result = mlflow.genai.evaluate(
                data=ds, predict_fn=predict_fn,
                model_id=m['agent_versions'][variant]['model_id'],
                scorers=[scorer(name=name)(fn) for name, fn in SCORERS.items()])
            filename = f'evaluation-native-{variant}.json'
            table = plain(result.result_df.to_dict(orient='records'))
            save_json(path.parent / filename, table)
            artifact = 'seed/evaluations/' + filename
            with mlflow.start_run(run_id=m['artifact_run_id']):
                mlflow.log_dict(table, artifact)
            completed[variant] = {
                'id': f"{m['seed_id']}-native-{variant}", 'variant': variant,
                'status': 'created', 'dataset_storage': 'native',
                'run_id': result.run_id, 'metrics': plain(result.metrics),
                'results_file': filename, 'results_artifact': artifact,
                'model_id': m['agent_versions'][variant]['model_id']}
            checkpoint()
        m.setdefault('previous_evaluations', m['evaluations'])
        m['evaluations'] = list(completed.values())
        active_ids = {e['id'] for e in m['evaluations']}
        for entry in m['traces']:
            if entry.get('evaluation') and entry['evaluation'] not in active_ids:
                entry['previous_evaluation'] = entry['evaluation']
                entry['evaluation'] = None
        mlflow.flush_trace_async_logging()
        expected = {r['id'] for r in m['dataset']['records']}
        for evaluation in m['evaluations']:
            persisted = mlflow.search_traces(locations=[m['experiment_id']],
                run_id=evaluation['run_id'], return_type='list', max_results=1000)
            covered = set()
            for trace in persisted:
                case_id = (trace.info.tags or {}).get('case_id')
                if case_id not in expected:
                    for span in trace.data.spans:
                        if isinstance(span.inputs, dict) and span.inputs.get('case_id') in expected:
                            case_id = span.inputs['case_id']
                            break
                if case_id in expected:
                    covered.add(case_id)
            if covered != expected:
                raise RuntimeError('Persisted evaluation traces do not yet cover every case; rerun to verify')
        return {'run_ids': [e['run_id'] for e in m['evaluations']], 'dataset': ds.name}
    attempt('Native evaluations', native_evaluations)

    def traffic():
        if any(not m.get('agent_versions', {}).get(v, {}).get('model_id') for v in ('baseline', 'improved')):
            return {'status': 'blocked', 'reason': 'Both agent model IDs must be created first.'}
        if db and not state.get('monitor_name'):
            raise RuntimeError('Online scorer must be configured before generating monitoring traffic')
        agent = build_agent(mlflow)
        entries = state.setdefault('monitoring_traces', [])
        done = state.setdefault('monitoring_cases', {})
        for idx, variant in enumerate(('baseline', 'improved')):
            for row in m['dataset']['records'][:3]:
                key = variant + ':' + row['id']
                if key not in done:
                    session_id = f"{m['seed_id']}-monitor-{variant}"
                    with mlflow.set_active_model(model_id=m['agent_versions'][variant]['model_id']):
                        with mlflow.start_span(name='monitored_support', span_type='AGENT') as span:
                            mlflow.update_current_trace(session_id=session_id,
                                tags={'monitoring_batch': m['seed_id'], 'synthetic': 'true'})
                            span.set_inputs(row['inputs'])
                            output = agent(row['id'], row['inputs']['messages'], variant,
                                           session_id, m['prompts'][idx])
                            span.set_outputs(output)
                        trace_id = mlflow.get_last_active_trace_id()
                    entries.append(trace_id)
                    done[key] = trace_id
                    m['traces'].append({'trace_id': trace_id, 'case_id': row['id'],
                        'variant': variant, 'session_id': session_id, 'evaluation': None})
                    checkpoint()
        mlflow.flush_trace_async_logging()
        for trace_id in entries:
            if mlflow.get_trace(trace_id) is None:
                raise RuntimeError('Monitoring trace not persisted')
        return {'trace_ids': entries}
    attempt('Monitoring traffic', traffic)
    if state.get('monitor_name') and state.get('monitoring_traces'):
        def check_monitoring():
            matches = []
            missing_tags = []
            for trace_id in state['monitoring_traces']:
                trace = mlflow.get_trace(trace_id)
                if (trace.info.tags or {}).get('monitoring_batch') != m['seed_id']:
                    missing_tags.append(trace_id)
                if successful_monitor_assessments(trace, state['monitor_name']):
                    matches.append(trace_id)
            if matches:
                return {'assessed_trace_ids': sorted(set(matches))}
            registered = {s.name: s for s in list_scorers(experiment_id=m['experiment_id'])}
            monitor = registered.get(state['monitor_name'])
            if monitor is None or not monitor.sample_rate:
                return {'status': 'blocked', 'reason': 'Online scorer is absent or stopped.'}
            if missing_tags:
                return {'status': 'blocked', 'reason': 'Monitoring traces lack the batch tag required by the scorer filter.',
                        'trace_ids': missing_tags}
            return {'status': 'pending', 'reason': 'Scorer is enabled and batch tags match, but no automatic assessments exist. Inspect scorer execution status in Databricks.',
                    'sample_rate': monitor.sample_rate, 'filter_string': monitor.filter_string}
        attempt('Online monitoring', check_monitoring)

    def reviews():
        state['review_operation'] = 'load_review_app'
        from databricks.agents.review_app import get_review_app
        state['review_operation'] = 'ensure_label_schema'
        app = get_review_app(experiment_id=m['experiment_id'])
        existing = [s.name for s in app.label_schemas if str(getattr(s, 'type', '')).lower().endswith('feedback')]
        schema_name = existing[0] if existing else ensure_review_schema(app, review_input(app))
        state['review_schema'] = schema_name
        checkpoint()
        if 'review_run_id' in state:
            state['review_operation'] = 'get_session'
            session = mlflow.genai.get_labeling_session(state['review_run_id'])
        else:
            name = 'synthetic-support-review-' + m['seed_id'][:8]
            state['review_operation'] = 'list_sessions'
            matches = [s for s in mlflow.genai.get_labeling_sessions() if s.name == name]
            state['review_operation'] = 'create_session'
            session = matches[0] if matches else mlflow.genai.create_labeling_session(
                name=name, assigned_users=[], label_schemas=[schema_name])
            state['review_run_id'] = session.mlflow_run_id
            checkpoint()
        if not state.get('review_traces_added'):
            state['review_operation'] = 'add_traces'
            ids = list(dict.fromkeys(t['trace_id'] for t in m['traces']))[:4]
            if not ids:
                raise RuntimeError('No traces to review')
            session.add_traces([mlflow.get_trace(t) for t in ids])
            state['review_traces_added'] = ids
            checkpoint()
        state['review_operation'] = 'read_back_session'
        session = mlflow.genai.get_labeling_session(state['review_run_id'])
        m['review_session'] = {'run_id': session.mlflow_run_id, 'url': session.url,
                               'trace_ids': state['review_traces_added']}
        return {**m['review_session'], 'status': 'manual',
                'reason': 'Open the review URL, configure a question if needed, and submit one review.'}
    if db:
        attempt('Reviews', reviews)
    else:
        report['Reviews'] = {'status': 'unsupported', 'reason': 'This extension uses Databricks Reviews.'}

    def traces():
        found = mlflow.search_traces(locations=[m['experiment_id']], max_results=100,
                                     return_type='list')
        if not found:
            raise RuntimeError('No persisted traces found')
        return {'sample_count': len(found), 'limit': 100}
    attempt('Traces', traces)
    report['Overview'] = {**report['Traces'], 'note': 'No real token costs: execution is scripted.'}

    def sessions():
        found = [mlflow.get_trace(t['trace_id']) for t in m['traces'][:24]]
        ids = {t.info.trace_metadata.get('mlflow.trace.session') for t in found}
        ids.discard(None)
        if not ids:
            raise RuntimeError('No persisted session metadata found')
        return {'sample_session_count': len(ids)}
    attempt('Sessions', sessions)

    def evaluations():
        runs = [client.get_run(e['run_id']) for e in m['evaluations'] if e.get('run_id')]
        if not runs:
            raise RuntimeError('No evaluation runs found')
        return {'run_ids': [r.info.run_id for r in runs]}
    attempt('Evaluation runs', evaluations)
    m['setup_complete'] = all(r['status'] == 'verified' for r in report.values())
    checkpoint()
    # Sync the updated source inventory for export; upload failure propagates.
    with mlflow.start_run(run_id=m['artifact_run_id']):
        mlflow.log_dict(m, 'seed/manifest.json')
        mlflow.log_dict(report, 'seed/tab-report.json')
    return m
