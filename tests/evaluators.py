"""Hand-written LangSmith equivalents for the three known demo scorers.

Use with Client.evaluate(target, data=dataset_name, evaluators=EVALUATORS).
This does not load or execute arbitrary scorer source from an export bundle.
"""
from stack_common import score_answer, score_tools, score_escalation


def answer_match(outputs, reference_outputs):
    return {'key': 'answer_match', 'score': float(score_answer(outputs, reference_outputs))}


def tool_sequence(outputs, reference_outputs):
    return {'key': 'tool_sequence', 'score': float(score_tools(outputs, reference_outputs))}


def escalation_match(outputs, reference_outputs):
    return {'key': 'escalation_match', 'score': float(score_escalation(outputs, reference_outputs))}


EVALUATORS = [answer_match, tool_sequence, escalation_match]
