import traceback
import pandas as pd
from utils.arize import arize_get_datasets, arize_get_dataset_examples
from utils.langsmith import ls_create_dataset, ls_upload_examples


def _extract_expected_value(record: dict) -> str:
    keys = [
        "expected",
        "expected_output",
        "expectedOutput",
        "reference_output",
        "referenceOutput",
        "output",
        "output.value",
    ]

    for k in keys:
        if k in record and record.get(k) not in (None, ""):
            return record.get(k)

    for v in record.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v and v.get(k) not in (None, ""):
                    return v.get(k)

    return ""


def arize_example_conversion(records: list[dict]) -> list[dict]:
    """Convert Arize dataset records to LangSmith format.
    
    Note: Dataset versioning and experiment associations are not migrated.
    """
    examples = []
    for r in records:
        inputs = {}

        for key, value in r.items():
            if key.startswith("input.") or key.startswith("input_"):
                clean_key = key.replace("input.", "").replace("input_", "")
                inputs[clean_key] = value
            elif key == "input":
                if isinstance(value, dict):
                    inputs.update(value)
                else:
                    inputs["input"] = value

        if not inputs:
            inputs = (
                r.get("inputs")
                or r.get("input")
                or {"input": r.get("prompt", r.get("question", ""))}
            )
            if not isinstance(inputs, dict):
                inputs = {"input": inputs}

        meta = r.get("metadata") or r.get("meta") or {}

        for key, value in r.items():
            if key.startswith("metadata.") or key.startswith("meta."):
                clean_key = key.replace("metadata.", "").replace("meta.", "")
                meta[clean_key] = value

        for mk in ("metadata", "meta"):
            if mk in inputs:
                val = inputs.pop(mk)
                if isinstance(val, dict):
                    meta = {**meta, **val}

        expected = _extract_expected_value(r)

        if not expected:
            expected = r.get("output.value", r.get("output", ""))

        ex = {
            "inputs": inputs,
            "outputs": {"reference_output": expected},
        }
        if isinstance(meta, dict) and meta:
            ex["metadata"] = meta
        examples.append(ex)
    return examples


def migrate_datasets(workspace_id: str):
    try:
        datasets_df = arize_get_datasets()
    except Exception as e:
        print(f"    x failed to fetch Arize datasets: {e}")
        traceback.print_exc()
        return

    if datasets_df is None or datasets_df.empty:
        print("    - no datasets found in Arize space")
        return

    for _, ds_row in datasets_df.iterrows():
        ds_id = ds_row.get("id") or ds_row.get("dataset_id")
        ds_name = ds_row.get("name") or f"arize-dataset-{ds_id}"
        print(f"    - migrating dataset: {ds_name}")
        try:
            ls_ds_id = ls_create_dataset(workspace_id, ds_name)
            
            if ls_ds_id is None:
                print(f"       • skipped (already exists)")
                continue

            examples_df = arize_get_dataset_examples(dataset_id=ds_id)
            
            if examples_df is not None and not examples_df.empty:
                records = examples_df.to_dict(orient="records")
                examples = arize_example_conversion(records)
                ls_upload_examples(workspace_id, ls_ds_id, examples)
                print(f"       • uploaded {len(examples)} examples")
            else:
                print(f"       • no examples found")
        except Exception as e:
            print(f"       x dataset '{ds_name}' failed: {e}")
            traceback.print_exc()
