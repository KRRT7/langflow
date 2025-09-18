from langflow.services.storage.constants import EXTENSION_TO_CONTENT_TYPE


def build_content_type_from_extension(extension: str):
    ext = extension.lower()
    if ext in EXTENSION_TO_CONTENT_TYPE:
        return EXTENSION_TO_CONTENT_TYPE[ext]
    return "application/octet-stream"
