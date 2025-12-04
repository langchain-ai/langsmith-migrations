import traceback
import json
from utils.phoenix import phoenix_get_datasets, phoenix_get_dataset_examples
from utils.langsmith import ls_create_dataset, ls_upload_examples


def _unpack_dotted_keys(record: dict) -> dict:
    """Unpack keys with dots into nested dicts.
    
    Example: {'output.tool_calls': '...'} -> {'output': {'tool_calls': '...'}}
    """
    result = {}
    for key, value in record.items():
        if '.' in key:
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    # If it's not a dict, wrap the existing value
                    current[part] = {"_value": current[part]}
                current = current[part]
            current[parts[-1]] = value
        else:
            if key in result and isinstance(result[key], dict):
                # Merge with existing dict
                if isinstance(value, dict):
                    result[key].update(value)
                else:
                    result[key]["_value"] = value
            else:
                result[key] = value
    return result


def _try_parse_json(value):
    """Try to parse a JSON string, return original if not JSON."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
    return value


def _extract_expected_value(record: dict) -> str:
    keys = [
        "expected",
        "expected_output",
        "expectedOutput",
        "reference_output",
        "referenceOutput",
        "output",
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


def phoenix_example_conversion(records: list) -> list[dict]:
    """Convert Phoenix dataset examples to LangSmith format."""
    examples = []
    for r in records:
        # Handle both dict and object types
        if hasattr(r, '__dict__'):
            r = vars(r) if not hasattr(r, 'to_dict') else r.to_dict()
        if not isinstance(r, dict):
            r = {"input": str(r)}
        
        # Unpack dotted keys into nested dicts
        r = _unpack_dotted_keys(r)
        
        # Parse any JSON strings in values
        for key in list(r.keys()):
            r[key] = _try_parse_json(r[key])
            if isinstance(r[key], dict):
                for k2 in list(r[key].keys()):
                    r[key][k2] = _try_parse_json(r[key][k2])
        
        # Inputs - handle list of messages
        inputs = r.get("input") or r.get("inputs") or {}
        if isinstance(inputs, list):
            # List of messages like [{"role": "user", "content": "..."}]
            inputs = {"messages": inputs}
        elif not isinstance(inputs, dict):
            inputs = {"input": inputs}

        # Metadata - merge from top-level and nested
        meta = r.get("metadata") or {}
        if isinstance(meta, dict):
            # Parse any JSON strings in metadata values
            for k in list(meta.keys()):
                meta[k] = _try_parse_json(meta[k])

        # Expected output - handle list of messages
        expected = r.get("output") or r.get("outputs") or _extract_expected_value(r)
        if isinstance(expected, list):
            # List of messages - extract content from assistant message or join all
            contents = []
            for msg in expected:
                if isinstance(msg, dict) and msg.get("content"):
                    contents.append(msg["content"])
                elif isinstance(msg, str):
                    contents.append(msg)
            expected = "\n".join(contents) if contents else str(expected)
        elif isinstance(expected, dict):
            expected = expected.get("content") or expected.get("output") or expected

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
        datasets = phoenix_get_datasets()
    except Exception as e:
        print(f"    x failed to fetch Phoenix datasets: {e}")
        traceback.print_exc()
        return

    if not datasets:
        print("    - no datasets found in Phoenix")
        return

    for ds in datasets:
        ds_id = ds.get("id")
        ds_name = ds.get("name") or f"phoenix-dataset-{ds_id}"
        print(f"    - migrating dataset: {ds_name}")
        try:
            ls_ds_id = ls_create_dataset(workspace_id, ds_name)
            
            if ls_ds_id is None:
                print(f"       • skipped (already exists)")
                continue

            examples = phoenix_get_dataset_examples(dataset_id=ds_id, dataset_name=ds_name)
            
            if examples:
                converted = phoenix_example_conversion(examples)
                ls_upload_examples(workspace_id, ls_ds_id, converted)
                print(f"       • uploaded {len(converted)} examples")
            else:
                print(f"       • no examples found")
        except Exception as e:
            print(f"       x dataset '{ds_name}' failed: {e}")
            traceback.print_exc()
