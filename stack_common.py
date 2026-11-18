"""Portable schema and deterministic demo fixtures. No third-party dependencies."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os

SCHEMA_VERSION = 1


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def plain(value):
    """Normalize SDK outputs, rejecting unsupported values rather than stringifying them."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'to_dictionary'):
        return plain(value.to_dictionary())
    if hasattr(value, 'to_dict'):
        return plain(value.to_dict())
    if hasattr(value, 'to_proto'):
        from google.protobuf.json_format import MessageToDict
        return plain(MessageToDict(value.to_proto(), preserving_proto_field_name=True))
    if hasattr(value, 'item'):
        return plain(value.item())
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    raise TypeError(f'Unsupported export type: {type(value).__name__}')


def fixtures():
    # Independently authored references. The agent implements its own factual logic.
    scenarios = [
        ('delivery', ['Where is DEMO-101?', 'Will it arrive before the weekend?', 'Do I need to sign?'],
         ['Your order is expected Friday.', 'Yes, delivery is expected Friday.', 'No signature is required.'],
         [['lookup_order', 'track_shipment'], ['track_shipment'], ['track_shipment']]),
        ('returns', ['Can I return headphones bought 12 days ago?', 'Even with an opened box?', 'Please start my return.'],
         ['Yes, the return window is 30 days.', 'Opened items are eligible.', 'A simulated return ticket was created; no real action was taken.'],
         [['check_return_eligibility', 'search_policy'], ['search_policy'], ['create_demo_ticket']]),
        ('billing', ['Why are there two charges?', 'Did I pay twice?', 'Please escalate this.'],
         ['One charge is posted and one is pending.', 'Only one charge has posted.', 'A simulated billing ticket was created; no real action was taken.'],
         [['lookup_payments', 'search_policy'], ['lookup_payments'], ['create_demo_ticket']]),
        ('retry', ['Track DEMO-202, please.', 'Where is it now?', 'Can someone follow up?'],
         ['The retry succeeded; delivery is expected Friday.', 'Your package is at the local depot.', 'A simulated delivery ticket was created; no real action was taken.'],
         [['lookup_order', 'track_shipment', 'track_shipment'], ['track_shipment'], ['create_demo_ticket']]),
    ]
    rows = []
    for scenario, questions, answers, tools in scenarios:
        history = [{'role': 'system', 'content': 'You are a fictional support assistant.'}]
        for turn, (question, expected, expected_tools) in enumerate(zip(questions, answers, tools)):
            history.append({'role': 'user', 'content': question})
            rows.append({
                'id': f'{scenario}-{turn}',
                'inputs': {'case_id': f'{scenario}-{turn}', 'messages': deepcopy(history)},
                'expectations': {'answer': expected, 'tools': expected_tools,
                                 'escalated': turn == 2 and scenario != 'delivery'},
                'metadata': {'scenario': scenario, 'turn': turn, 'synthetic': True},
            })
            history.append({'role': 'assistant', 'content': expected})
    return rows


def score_answer(outputs, expectations):
    return outputs['answer'].strip().casefold() == expectations['answer'].strip().casefold()


def score_tools(outputs, expectations):
    # Order and repetition matter: the retry must call track_shipment twice.
    return outputs['tools'] == expectations['tools']


def score_escalation(outputs, expectations):
    return outputs['escalated'] == expectations['escalated']


SCORERS = {'answer_match': score_answer, 'tool_sequence': score_tools,
           'escalation_match': score_escalation}
PROMPTS = [
    {'version': '1', 'template': 'Answer this support question: {{question}}', 'aliases': ['baseline']},
    {'version': '2', 'template': 'Use verified demo facts. Retry transient failures and escalate when requested. Question: {{question}}', 'aliases': ['production']},
]


def is_databricks_uri(uri):
    return uri == 'databricks' or uri.startswith('databricks://')


def databricks_registry_uri(tracking_uri):
    if not is_databricks_uri(tracking_uri):
        raise ValueError('A Databricks tracking URI is required')
    return tracking_uri.replace('databricks', 'databricks-uc', 1)
