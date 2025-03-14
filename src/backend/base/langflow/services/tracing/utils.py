from typing import Any

from langflow.schema.data import Data
from langflow.schema.message import Message


def convert_to_langchain_type(value):
    """Recursively converts values in the input to the appropriate langchain type."""
    cache = {}  # Cache to store the interim results to avoid redundant computation

    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, Message):
            if value in cache:
                return cache[value]

            if "prompt" in value:
                result = value.load_lc_prompt()
            elif value.sender:
                result = value.to_lc_message()
            else:
                result = value.to_lc_document()

            cache[value] = result
            return result
        if isinstance(value, Data):
            if value in cache:
                return cache[value]

            result = value.to_lc_document() if "text" in value.data else value.data
            cache[value] = result
            return result
        return value

    return _convert(value)


def convert_to_langchain_types(io_dict: dict[str, Any]):
    """Converts all values in the given dictionary to langchain types."""
    return {key: convert_to_langchain_type(value) for key, value in io_dict.items()}
