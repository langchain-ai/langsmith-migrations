import traceback
import pandas as pd
from utils.arize import arize_get_datasets, arize_get_dataset_examples
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
        "output",
        "output.value",
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


def arize_example_conversion(records: list[dict]) -> list[dict]:
    """Convert Arize dataset records to LangSmith format.

    Arize datasets may have various column naming conventions:
    - input.value, output.value (dotted notation from OpenInference)
    - input, output (simple)
    - Various metadata columns
    """
    examples = []
    for r in records:
        # Handle Arize's dotted notation (input.value, input.context, etc.)
        inputs = {}

        # Check for dotted notation keys (common in Arize/OpenInference)
        for key, value in r.items():
            if key.startswith("input.") or key.startswith("input_"):
                # Convert input.value -> value, input.context -> context
                clean_key = key.replace("input.", "").replace("input_", "")
                inputs[clean_key] = value
            elif key == "input":
                if isinstance(value, dict):
                    inputs.update(value)
                else:
                    inputs["input"] = value

        # If no inputs found, check for common Arize keys
        if not inputs:
            inputs = (
                r.get("inputs")
                or r.get("input")
                or {"input": r.get("prompt", r.get("question", ""))}
            )
            if not isinstance(inputs, dict):
                inputs = {"input": inputs}

        # Move any accidental metadata fields from inputs into example metadata
        meta = r.get("metadata") or r.get("meta") or {}

        # Extract metadata from dotted notation
        for key, value in r.items():
            if key.startswith("metadata.") or key.startswith("meta."):
                clean_key = key.replace("metadata.", "").replace("meta.", "")
                meta[clean_key] = value

        for mk in ("metadata", "meta"):
            if mk in inputs:
                val = inputs.pop(mk)
                if isinstance(val, dict):
                    meta = {**meta, **val}

        # Expected/reference output
        expected = _extract_expected_value(r)

        # Also check for output.value in Arize format
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
    """Migrate all datasets from Arize to LangSmith.

    Args:
        workspace_id: LangSmith workspace ID to migrate datasets into.
    """
    try:
        datasets_df = arize_get_datasets()
    except Exception as e:
        print(f"    x failed to fetch Arize datasets: {e}")
        traceback.print_exc()
        return

    if datasets_df is None or datasets_df.empty:
        print("    - no datasets found in Arize space")
        return

    # Iterate over DataFrame rows
    for _, ds_row in datasets_df.iterrows():
        ds_id = ds_row.get("id") or ds_row.get("dataset_id")
        ds_name = ds_row.get("name") or f"arize-dataset-{ds_id}"
        print(f"    - migrating dataset: {ds_name}")
        try:
            # Create dataset in LangSmith (returns None if already exists)
            ls_ds_id = ls_create_dataset(workspace_id, ds_name)
            
            if ls_ds_id is None:
                print(f"       • skipped (already exists)")
                continue

            # Fetch examples from Arize (returns DataFrame)
            examples_df = arize_get_dataset_examples(dataset_id=ds_id)
            
            if examples_df is not None and not examples_df.empty:
                # Convert DataFrame to list of dicts for processing
                records = examples_df.to_dict(orient="records")
                examples = arize_example_conversion(records)
                ls_upload_examples(workspace_id, ls_ds_id, examples)
                print(f"       • uploaded {len(examples)} examples")
            else:
                print(f"       • no examples found")
        except Exception as e:
            print(f"       x dataset '{ds_name}' failed: {e}")
            traceback.print_exc()
