import json
from typing import Any

from langflow.schema.data import Data


def infer_list_type(items: list, max_samples: int = 5) -> str:
    """Infer the type of a list by sampling its items.

    Handles mixed types and provides more detailed type information.
    """
    if not items:
        return "list(unknown)"

    samples = items[:max_samples]
    n = len(samples)

    # Fast path: check types during iteration; exit early if mixed type found
    first_type = get_type_str(samples[0])
    all_same = True
    for i in range(1, n):
        t = get_type_str(samples[i])
        if t != first_type:
            all_same = False
            break

    if all_same:
        return f"list({first_type})"

    # Slow path: need to compute set (as much as Counter.keys())
    seen = set()
    for item in samples:
        seen.add(get_type_str(item))
    type_str = "|".join(sorted(seen))
    return f"list({type_str})"


def get_type_str(value: Any) -> str:
    """Get a detailed string representation of the type of a value.

    Handles special cases and provides more specific type information.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        val_lower = value.lower()
        date_patterns = ("date", "time", "yyyy", "mm/dd", "dd/mm", "yyyy-mm")
        # Faster: use any() over tuple
        if any(pat in val_lower for pat in date_patterns):
            return "str(possible_date)"
        # Check if it's a JSON string
        try:
            json.loads(value)
            return "str(json)"
        except (json.JSONDecodeError, TypeError):
            pass
        return "str"
    # Only convert to list if necessary
    if isinstance(value, (list, tuple, set)):
        return infer_list_type(value if isinstance(value, list) else list(value))
    if isinstance(value, dict):
        return "dict"
    # Handle custom objects
    return type(value).__name__


def analyze_value(
    value: Any,
    max_depth: int = 10,
    current_depth: int = 0,
    path: str = "",
    *,
    size_hints: bool = True,
    include_samples: bool = True,
) -> str | dict:
    """Analyze a value and return its structure with additional metadata.

    Args:
        value: The value to analyze
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        path: Current path in the structure
        size_hints: Whether to include size information for collections
        include_samples: Whether to include sample structure for lists
    """
    if current_depth >= max_depth:
        return f"max_depth_reached(depth={max_depth})"

    try:
        # Precompute for efficiency
        is_sequence = isinstance(value, (list, tuple, set))
        if is_sequence:
            length = len(value)
            if length == 0:
                return "list(unknown)"

            type_info = infer_list_type(value if isinstance(value, list) else list(value))
            size_info = f"[size={length}]" if size_hints else ""

            # Compound condition precomputed
            is_sampled_item_complex = (
                include_samples
                and length > 0
                and isinstance(value, (list, tuple))
                and isinstance(value[0], (dict, list))
                and current_depth < max_depth - 1
            )
            if is_sampled_item_complex:
                sample = analyze_value(
                    value[0],
                    max_depth,
                    current_depth + 1,
                    f"{path}[0]",
                    size_hints=size_hints,
                    include_samples=include_samples,
                )
                return f"{type_info}{size_info}, sample: {json.dumps(sample)}"

            return f"{type_info}{size_info}"

        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                new_path = f"{path}.{k}" if path else k
                try:
                    result[k] = analyze_value(
                        v,
                        max_depth,
                        current_depth + 1,
                        new_path,
                        size_hints=size_hints,
                        include_samples=include_samples,
                    )
                except Exception as e:  # noqa: BLE001
                    result[k] = f"error({e!s})"
            return result

        return get_type_str(value)

    except Exception as e:  # noqa: BLE001
        return f"error({e!s})"


def get_data_structure(
    data_obj: Data | dict,
    max_depth: int = 10,
    max_sample_size: int = 3,
    *,
    size_hints: bool = True,
    include_sample_values: bool = False,
    include_sample_structure: bool = True,
) -> dict:
    """Convert a Data object or dictionary into a detailed schema representation.

    Args:
        data_obj: The Data object or dictionary to analyze
        max_depth: Maximum depth for nested structures
        size_hints: Include size information for collections
        include_sample_values: Whether to include sample values in the output
        include_sample_structure: Whether to include sample structure for lists
        max_sample_size: Maximum number of sample values to include

    Returns:
        dict: A dictionary containing:
            - structure: The structure of the data
            - samples: (optional) Sample values from the data

    Example:
        >>> data = {
        ...     "name": "John",
        ...     "scores": [1, 2, 3, 4, 5],
        ...     "details": {
        ...         "age": 30,
        ...         "cities": ["NY", "LA", "SF", "CHI"],
        ...         "metadata": {
        ...             "created": "2023-01-01",
        ...             "tags": ["user", "admin", 123]
        ...         }
        ...     }
        ... }
        >>> result = get_data_structure(data)
        {
            "structure": {
                "name": "str",
                "scores": "list(int)[size=5]",
                "details": {
                    "age": "int",
                    "cities": "list(str)[size=4]",
                    "metadata": {
                        "created": "str(possible_date)",
                        "tags": "list(str|int)[size=3]"
                    }
                }
            }
        }
    """
    # Handle both Data objects and dictionaries
    data = data_obj.data if isinstance(data_obj, Data) else data_obj

    result = {
        "structure": analyze_value(
            data, max_depth=max_depth, size_hints=size_hints, include_samples=include_sample_structure
        )
    }

    if include_sample_values:
        result["samples"] = get_sample_values(data, max_items=max_sample_size)

    return result


def get_sample_values(data: Any, max_items: int = 3) -> Any:
    """Get sample values from a data structure, handling nested structures."""
    if isinstance(data, list | tuple | set):
        return [get_sample_values(item) for item in list(data)[:max_items]]
    if isinstance(data, dict):
        return {k: get_sample_values(v, max_items) for k, v in data.items()}
    return data
