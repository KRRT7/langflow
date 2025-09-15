from typing import Any

from lfx.schema.data import Data

from langflow.schema.message import Message


def convert_to_langchain_type(value):
    if type(value) is dict:
        value = {key: convert_to_langchain_type(val) for key, val in value.items()}
    elif type(value) is list:
        value = [convert_to_langchain_type(v) for v in value]
    elif type(value) is Message:
        if "prompt" in value:
            value = value.load_lc_prompt()
        elif value.sender:
            value = value.to_lc_message()
        else:
            value = value.to_lc_document()
    elif type(value) is Data:
        value = value.to_lc_document() if "text" in value.data else value.data
    return value


def convert_to_langchain_types(io_dict: dict[str, Any]):
    return {key: convert_to_langchain_type(value) for key, value in io_dict.items()}
