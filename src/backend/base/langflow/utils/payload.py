import re
from typing import Any


def extract_input_variables(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extracts input variables from the template and adds them to the input_variables field."""
    prompt_pattern = re.compile(r"\{(.*?)\}")

    for node in nodes:
        try:
            data_node = node["data"]["node"]
            template_info = data_node["template"]
            template_type = template_info["_type"]

            if "input_variables" in template_info:
                if template_type == "prompt":
                    value = template_info["template"]["value"]
                    variables = prompt_pattern.findall(value)
                elif template_type == "few_shot":
                    prefix = template_info["prefix"]["value"]
                    suffix = template_info["suffix"]["value"]
                    variables = prompt_pattern.findall(prefix + suffix)
                else:
                    variables = []

                template_info["input_variables"]["value"] = variables
        except (KeyError, TypeError):
            # Exception suppressed as in the original code
            pass

    return nodes


def get_root_vertex(graph):
    """Returns the root node of the template."""
    incoming_edges = {edge.source_id for edge in graph.edges}

    if not incoming_edges and len(graph.vertices) == 1:
        return graph.vertices[0]

    return next((node for node in graph.vertices if node.id not in incoming_edges), None)


def build_json(root, graph) -> dict:
    if "node" not in root.data:
        # If the root node has no "node" key, then it has only one child,
        # which is the target of the single outgoing edge
        edge = root.edges[0]
        local_nodes = [edge.target]
    else:
        # Otherwise, find all children whose type matches the type
        # specified in the template
        node_type = root.node_type
        local_nodes = graph.get_nodes_with_target(root)

    if len(local_nodes) == 1:
        return build_json(local_nodes[0], graph)
    # Build a dictionary from the template
    template = root.data["node"]["template"]
    final_dict = template.copy()

    for key in final_dict:
        if key == "_type":
            continue

        value = final_dict[key]
        node_type = value["type"]

        if "value" in value and value["value"] is not None:
            # If the value is specified, use it
            value = value["value"]
        elif "dict" in node_type:
            # If the value is a dictionary, create an empty dictionary
            value = {}
        else:
            # Otherwise, recursively build the child nodes
            children = []
            for local_node in local_nodes:
                node_children = graph.get_children_by_node_type(local_node, node_type)
                children.extend(node_children)

            if value["required"] and not children:
                msg = f"No child with type {node_type} found"
                raise ValueError(msg)
            values = [build_json(child, graph) for child in children]
            value = (
                list(values) if value["list"] else next(iter(values), None)  # type: ignore[arg-type]
            )
        final_dict[key] = value

    return final_dict


def extract_variables(template: dict[str, Any], pattern: re.Pattern) -> list[str]:
    """Helper function to extract variables based on the provided pattern."""
    return pattern.findall(template)
