"""Langfuse migration: trace mapping, v2 observations reader, and prompt tagging.

No credentials or network: Langfuse and LangSmith calls are patched.
"""

import datetime as dt
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from langsmith.utils import LangSmithConflictError

from providers.langfuse.data import prompts, traces
from providers.langfuse.data.prompts import build_chat_prompt, dotted_variables, langsmith_prompt_name
from providers.langfuse.data.traces import (
    _fit_into_ingest_window,
    _usage_metadata,
    map_langfuse_to_langsmith,
    trace_from_observations,
)
from utils import langsmith as ls_utils


def _gen(**kw):
    return {"type": "GENERATION", **kw}


class UsageMetadataTest(unittest.TestCase):
    def test_anthropic_cache_tokens_count_as_input(self):
        out = _usage_metadata(_gen(
            usageDetails={"input": 4, "output": 107, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 6908, "total": 7019},
            costDetails={"input": 0.0008639, "output": 0.0000535, "total": 0.0009174},
            totalCost=0.0009174,
        ))
        self.assertEqual(out["input_tokens"], 6912)
        self.assertEqual(out["output_tokens"], 107)
        self.assertEqual(out["total_tokens"], 7019)
        self.assertEqual(out["input_token_details"], {"cache_creation": 6908})
        self.assertAlmostEqual(out["input_cost"], 0.0008639)
        self.assertAlmostEqual(out["output_cost"], 0.0000535)
        self.assertAlmostEqual(out["total_cost"], 0.0009174)

    def test_without_total_sums_the_input_keys(self):
        out = _usage_metadata(_gen(usageDetails={"input": 10, "input_cached_tokens": 5, "output": 3,
                                                 "output_reasoning": 2}))
        self.assertEqual(out["input_tokens"], 15)
        self.assertEqual(out["output_tokens"], 5)
        self.assertEqual(out["total_tokens"], 20)
        self.assertEqual(out["input_token_details"], {"cache_read": 5})
        self.assertEqual(out["output_token_details"], {"reasoning": 2})

    def test_v3_trace_detail_fields(self):
        out = _usage_metadata(_gen(usage={"input": 10, "output": 5, "total": 15, "unit": "TOKENS"},
                                   calculatedInputCost=0.1, calculatedOutputCost=0.2,
                                   calculatedTotalCost=0.3))
        self.assertEqual((out["input_tokens"], out["output_tokens"], out["total_tokens"]), (10, 5, 15))
        self.assertEqual((out["input_cost"], out["output_cost"], out["total_cost"]), (0.1, 0.2, 0.3))

    def test_no_usage(self):
        self.assertIsNone(_usage_metadata(_gen()))


def _obs(id, parent, type_, name, start, end, **kw):
    return {"id": id, "traceId": "t1", "parentObservationId": parent, "type": type_, "name": name,
            "startTime": start, "endTime": end, **kw}


V2_TRACE = [
    _obs("g1", "r1", "GENERATION", "claude", "2026-11-20T10:00:00.100Z", "2026-11-20T10:00:02.000Z",
         model="claude-haiku-5-5", input=json.dumps([{"role": "user", "content": "hi"}]),
         output=json.dumps({"role": "assistant", "content": "hello"}),
         usageDetails={"input": 3, "output": 2, "total": 5}, sessionId="s1", userId="u1",
         metadata={"plan": "team", "resourceAttributes.service.name": "x", "attributes.user.id": "u1"}),
    _obs("r1", None, "SPAN", "chat", "2026-11-20T10:00:00.000Z", "2026-11-20T10:00:02.000Z",
         input=json.dumps({"question": "hi"}), output="plain text answer", sessionId="s1",
         userId="u1", tags=["plan:team"], release="app@1", metadata={"plan": "team"}),
]


class TraceFromObservationsTest(unittest.TestCase):
    def test_single_root_span_becomes_root_run(self):
        trace = trace_from_observations(V2_TRACE)
        self.assertEqual(trace["name"], "chat")
        self.assertEqual(trace["input"], {"question": "hi"})
        self.assertEqual(trace["output"], "plain text answer")
        self.assertEqual([o["id"] for o in trace["observations"]], ["g1"])
        self.assertEqual(trace["metadata"], {"plan": "team", "release": "app@1"})

        runs = map_langfuse_to_langsmith(trace)
        root, llm = runs
        self.assertEqual(llm["parent_run_id"], root["id"])
        self.assertEqual(llm["inputs"], {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(llm["metadata"]["ls_model_name"], "claude-haiku-5-5")
        self.assertNotIn("resourceAttributes.service.name", llm["metadata"])
        self.assertEqual(root["metadata"]["session_id"], "s1")
        self.assertEqual(root["metadata"]["user_id"], "u1")
        self.assertEqual(root["metadata"]["langfuse_trace_id"], "t1")
        self.assertEqual(root["tags"], ["plan:team"])
        self.assertTrue(llm["dotted_order"].startswith(root["dotted_order"] + "."))

    def test_several_roots_hang_off_a_trace_run(self):
        second = _obs("r2", None, "SPAN", "other", "2026-11-20T10:00:03Z", "2026-11-20T10:00:04Z")
        trace = trace_from_observations(V2_TRACE + [second])
        self.assertEqual(len(trace["observations"]), 3)
        runs = map_langfuse_to_langsmith(trace)
        root_id = runs[0]["id"]
        self.assertEqual(sum(r["parent_run_id"] == root_id for r in runs[1:]), 2)

    def test_error_level_becomes_run_error(self):
        failed = [dict(o, level="ERROR", statusMessage="429 rate limited") for o in V2_TRACE]
        runs = map_langfuse_to_langsmith(trace_from_observations(failed))
        self.assertEqual([r["error"] for r in runs], ["429 rate limited"] * 2)

    def test_ingest_window_keeps_order_durations_and_ends_before_now(self):
        long_trace = [dict(o, traceId="t2", startTime=o["startTime"].replace("10:00", "11:00"),
                           endTime="2026-11-20T11:05:00.000Z") for o in V2_TRACE]
        batch = [map_langfuse_to_langsmith(trace_from_observations(t)) for t in (V2_TRACE, long_trace)]
        now = dt.datetime(2026, 11, 24, 12, 0, tzinfo=dt.timezone.utc)
        _fit_into_ingest_window(batch, now)
        ts = lambda v: dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        (root1, llm1), (root2, _) = batch
        self.assertLessEqual(ts(root2["end_time"]), now)          # 5-minute trace ends in the past
        self.assertEqual(ts(root2["end_time"]) - ts(root2["start_time"]), dt.timedelta(minutes=5))
        self.assertLess(ts(root1["end_time"]), ts(root2["start_time"]))  # order kept, no overlap
        self.assertEqual(ts(llm1["start_time"]) - ts(root1["start_time"]), dt.timedelta(milliseconds=100))
        self.assertTrue(llm1["dotted_order"].startswith(root1["dotted_order"] + "."))
        self.assertEqual(root1["metadata"]["langfuse_start_time"], "2026-11-20T10:00:00.000Z")


class FetchTracesTest(unittest.TestCase):
    def _run(self, v2: bool):
        legacy_detail = {"id": "old", "name": "chat", "timestamp": "2026-11-20T10:00:00Z",
                         "input": "q", "output": "a", "observations": []}
        with mock.patch.object(traces, "NUM_TRACES_TO_REPLAY", "5"), \
             mock.patch.object(traces, "lf_supports_observations_v2", return_value=v2), \
             mock.patch.object(traces, "lf_recent_trace_ids", return_value=["t1"]), \
             mock.patch.object(traces, "lf_get_trace_observations", return_value=V2_TRACE), \
             mock.patch.object(traces, "_legacy_recent_trace_ids", return_value=["old"]), \
             mock.patch.object(traces, "lf_get", return_value=legacy_detail) as lf_get, \
             mock.patch.object(traces, "ls_replay_runs_sdk") as replay, \
             mock.patch("builtins.print"):
            traces.fetch_and_transform_traces("ws", sleep_between_gets=0)
        return replay.call_args.args[1], lf_get

    def test_v2_observations_api(self):
        runs, lf_get = self._run(v2=True)
        self.assertEqual(runs[0]["metadata"]["langfuse_trace_id"], "t1")
        lf_get.assert_not_called()

    def test_falls_back_to_v3_traces_api(self):
        runs, lf_get = self._run(v2=False)
        lf_get.assert_called_once_with("/api/public/traces/old")
        self.assertEqual(runs[0]["metadata"]["langfuse_trace_id"], "old")

    def test_old_traces_are_moved_into_the_ingest_window(self):
        runs, _ = self._run(v2=True)
        start = dt.datetime.fromisoformat(runs[0]["start_time"].replace("Z", "+00:00"))
        self.assertLess(dt.datetime.now(dt.timezone.utc) - start, dt.timedelta(hours=1))


class CommitTagsTest(unittest.TestCase):
    def test_tags_are_sent_with_a_string_commit_id(self):
        commit = SimpleNamespace(commit_hash="abcd1234ef", id=uuid.uuid4(), owner="-", repo="p")
        client = mock.Mock()
        client.push_prompt.return_value = "https://smith/prompts/p/abcd1234?organizationId=x"
        client.list_prompt_commits.return_value = [commit]

        def create(name, commit_id, tag):
            json.dumps({"commit_id": commit_id})  # the SDK serializes the id as JSON
            if tag == "bad":
                raise RuntimeError("409 conflict")

        client._create_commit_tags.side_effect = create
        with mock.patch.object(ls_utils, "_ls_client", return_value=client):
            _, applied, failed, unchanged = ls_utils.ls_push_prompt_version(
                "p", "", object(), "ws", commit_tags=["production", "bad"])
        self.assertEqual(applied, ["production"])
        self.assertEqual(failed, [("bad", "409 conflict")])
        self.assertFalse(unchanged)
        self.assertIsInstance(client._create_commit_tags.call_args.args[1], str)

    def test_unchanged_version_tags_the_previous_commit(self):
        previous = SimpleNamespace(commit_hash="0123abcdef", id=uuid.uuid4(), owner="-", repo="p")
        client = mock.Mock()
        client.push_prompt.side_effect = LangSmithConflictError("409 Nothing to commit")
        client.list_prompt_commits.return_value = [previous]
        with mock.patch.object(ls_utils, "_ls_client", return_value=client):
            url, applied, failed, unchanged = ls_utils.ls_push_prompt_version(
                "p", "", object(), "ws", commit_tags=["staging"])
        self.assertTrue(unchanged)
        self.assertEqual((url, applied, failed), ("p:0123abcd", ["staging"], []))
        self.assertEqual(client._create_commit_tags.call_args.args[1], str(previous.id))

    def test_other_conflicts_are_raised(self):
        client = mock.Mock()
        client.push_prompt.side_effect = LangSmithConflictError(
            "409 Parent commit validation failed: reference ID x")
        with mock.patch.object(ls_utils, "_ls_client", return_value=client):
            with self.assertRaises(LangSmithConflictError):
                ls_utils.ls_push_prompt_version("p", "", object(), "ws", commit_tags=["staging"])
        client._create_commit_tags.assert_not_called()


class PromptConversionTest(unittest.TestCase):
    def test_variable_values_are_not_html_escaped(self):
        value = '<doc a="1">x & y</doc>'
        chat = build_chat_prompt({"prompt": [{"role": "system", "content": 'JSON: {"k": 1}'},
                                             {"role": "user", "content": "ctx: {{ ctx }}"}]})
        msgs = chat.invoke({"ctx": value}).to_messages()
        self.assertEqual(msgs[0].content, 'JSON: {"k": 1}')
        self.assertEqual(msgs[1].content, f"ctx: {value}")
        text = build_chat_prompt({"prompt": "{{a}} and {{{b}}} {{#s}}x{{/s}}"})
        self.assertEqual(text.messages[0].prompt.template, "{{{a}}} and {{{b}}} {{#s}}x{{/s}}")
        hyphen = build_chat_prompt({"prompt": "Hi {{user-name}}"})
        self.assertEqual(hyphen.invoke({"user-name": "A&B"}).to_messages()[0].content, "Hi A&B")

    def test_dotted_variables_are_reported(self):
        self.assertEqual(dotted_variables({"prompt": [{"role": "user", "content": "{{user.name}} {{q}}"}]}),
                         ["user.name"])
        self.assertEqual(dotted_variables({"prompt": "{{q}}"}), [])

    def test_prompt_names(self):
        self.assertEqual(langsmith_prompt_name("Team/Summarize.v2"), "team-summarize-v2")
        self.assertEqual(langsmith_prompt_name("qa:eu"), "qa-eu")
        self.assertEqual(langsmith_prompt_name("My Prompt_1"), "my-prompt_1")

    def test_name_collision_skips_the_second_prompt(self):
        history = [{"name": n, "versions": [{"version": 1, "prompt": "hi", "labels": []}]}
                   for n in ("team/foo", "team-foo")]
        with mock.patch.object(prompts, "lf_get_prompt_history", return_value=history), \
             mock.patch.object(prompts, "ls_prompt_exists", return_value=False), \
             mock.patch.object(prompts, "ls_push_prompt_version",
                               return_value=("url", [], [], False)) as push, \
             mock.patch("builtins.print") as out:
            prompts.migrate_prompts("ws", "proj")
        self.assertEqual(push.call_count, 1)
        self.assertIn("also the name of 'team/foo'", " ".join(str(c) for c in out.call_args_list))


if __name__ == "__main__":
    unittest.main()
