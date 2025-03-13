from pathlib import Path


def raw_frontend_data_is_valid(raw_frontend_data):
    """Check if the raw frontend data is valid for processing."""
    return "template" in raw_frontend_data and "display_name" in raw_frontend_data


def get_file_path_value(file_path):
    """Get the file path value if the file exists, else return empty string."""
    # Creating Path object and validating the basic correctness
    try:
        path = Path(file_path)
    except TypeError:
        return ""

    # Normalizing the path and checking its existence within the cache directory
    if not path.is_absolute() or not path.is_file() or not str(path).startswith(cache_dir):
        return ""

    return file_path if path.exists() else ""


def update_template_field(new_template, key, previous_value_dict) -> None:
    """Updates a specific field in the frontend template."""
    template_field = new_template.get(key)

    # Validate type matching and existence of template_field
    if not template_field or template_field.get("type") != previous_value_dict.get("type"):
        return

    previous_value = previous_value_dict.get("value", None)
    previous_file_path = previous_value_dict.get("file_path", "")

    if previous_value is not None:
        if template_field.get("value") != previous_value:
            template_field["load_from_db"] = previous_value_dict.get("load_from_db", False)
        template_field["value"] = previous_value

    if previous_file_path:
        file_path_value = get_file_path_value(previous_file_path)
        if not file_path_value:
            template_field["value"] = ""
        template_field["file_path"] = file_path_value


def is_valid_data(frontend_node, raw_frontend_data):
    """Check if the data is valid for processing."""
    return frontend_node and "template" in frontend_node and raw_frontend_data_is_valid(raw_frontend_data)


def update_template_values(new_template, previous_template) -> None:
    """Updates the frontend template with values from the raw template."""
    for key, previous_value_dict in previous_template.items():
        if key == "code" or not isinstance(previous_value_dict, dict):
            continue

        update_template_field(new_template, key, previous_value_dict)


def update_frontend_node_with_template_values(frontend_node, raw_frontend_node):
    """Updates the given frontend node with values from the raw template data.

    :param frontend_node: A dict representing a built frontend node.
    :param raw_frontend_node: A dict representing raw template data.
    :return: Updated frontend node.
    """
    if not is_valid_data(frontend_node, raw_frontend_node):
        return frontend_node

    update_template_values(frontend_node["template"], raw_frontend_node["template"])

    old_code = raw_frontend_node["template"]["code"]["value"]
    new_code = frontend_node["template"]["code"]["value"]
    frontend_node["edited"] = raw_frontend_node.get("edited", False) or (old_code != new_code)

    # Compute tool modes from template
    tool_modes = [
        value.get("tool_mode")
        for key, value in frontend_node["template"].items()
        if key != "_type" and isinstance(value, dict)
    ]

    if any(tool_modes):
        frontend_node["tool_mode"] = raw_frontend_node.get("tool_mode", False)
    else:
        frontend_node["tool_mode"] = False

    if not frontend_node.get("edited", False):
        frontend_node["display_name"] = raw_frontend_node.get("display_name", frontend_node.get("display_name", ""))
        frontend_node["description"] = raw_frontend_node.get("description", frontend_node.get("description", ""))

    return frontend_node
