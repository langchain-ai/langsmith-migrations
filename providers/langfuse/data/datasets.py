import traceback
from utils.langfuse import lf_get_project_datasets
from utils.langsmith import ls_create_dataset, ls_upload_examples


def _extract_expected_value(record: dict) -> str:
    """Extract expected/reference output from a record using common keys.

    Checks both snake_case and camelCase at top-level and one nested dict.
    """
    keys = [
        "expected",
        "expected_output",
        "expectedOutput",
        "reference_output",
        "referenceOutput",
    ]

    # Direct keys
    for k in keys:
        if k in record and record.get(k) not in (None, ""):
            return record.get(k)

    # One-level nested dict
    for v in record.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v and v.get(k) not in (None, ""):
                    return v.get(k)

    return ""


def langfuse_example_conversion(records: list[dict]) -> list[dict]:
    """Convert Langfuse dataset records to LangSmith format."""
    examples = []
    for r in records:
        # Inputs: prefer dict, fallback to scalar under "input"
        inputs = (
            r.get("inputs")
            or r.get("input")
            or {"input": r.get("prompt", "")}
        )
        if not isinstance(inputs, dict):
            inputs = {"input": inputs}

        # Move any accidental metadata fields from inputs into example metadata
        meta = r.get("metadata") or r.get("meta") or {}
        for mk in ("metadata", "meta"):
            if mk in inputs:
                val = inputs.pop(mk)
                if isinstance(val, dict):
                    meta = {**meta, **val}

        # Expected/reference output
        expected = _extract_expected_value(r)

        ex = {
            "inputs": inputs,
            "outputs": {"reference_output": expected},
        }
        if isinstance(meta, dict) and meta:
            ex["metadata"] = meta
        examples.append(ex)
    return examples


def migrate_datasets(workspace_id: str, project_id: str):
    ds_map = lf_get_project_datasets(project_id)
    for ds_name, records in ds_map.items():
        print(f"    - migrating dataset: {ds_name}")
        try:
            ds_id = ls_create_dataset(workspace_id, ds_name)
            if records:
                examples = langfuse_example_conversion(records)
                ls_upload_examples(workspace_id, ds_id, examples)
                print(f"       • uploaded {len(examples)} examples")
        except Exception as e:
            print(f"       x dataset '{ds_name}' failed: {e}")
            traceback.print_exc()