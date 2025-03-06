import contextlib
import re


def extract_input_variables(nodes):
    """Extracts input variables from the template and adds them to the input_variables field."""
    for node in nodes:
        with contextlib.suppress(Exception):
            if "input_variables" in node["data"]["node"]["template"]:
                if node["data"]["node"]["template"]["_type"] == "prompt":
                    variables = re.findall(
                        r"\{(.*?)\}",
                        node["data"]["node"]["template"]["template"]["value"],
                    )
                elif node["data"]["node"]["template"]["_type"] == "few_shot":
                    variables = re.findall(
                        r"\{(.*?)\}",
                        node["data"]["node"]["template"]["prefix"]["value"]
                        + node["data"]["node"]["template"]["suffix"]["value"],
                    )
                else:
                    variables = []
                node["data"]["node"]["template"]["input_variables"]["value"] = variables
    return nodes


def get_root_vertex(graph):
    """Returns the root node of the template."""
    incoming_edges = {edge.source_id for edge in graph.edges}

    if not incoming_edges and len(graph.vertices) == 1:
        return graph.vertices[0]

    return next((node for node in graph.vertices if node.id not in incoming_edges), None)


def build_json(root, graph) -> dict:
    """Function to build a JSON description of nodes and their relationships from a root node.
    """

    def get_local_nodes(root, graph):
        if "node" not in root.data:
            # Single child scenario
            return [root.edges[0].target]
        # Multiple local nodes
        return graph.get_nodes_with_target(root)

    def recursive_build_json(local_nodes, graph):
        if len(local_nodes) == 1:
            return build_json(local_nodes[0], graph)

        root = local_nodes[0]
        template = root.data["node"]["template"]
        final_dict = template.copy()

        for key in template:
            if key == "_type":
                continue

            value = template[key]
            node_type = value["type"]

            if "value" in value and value["value"] is not None:
                final_dict[key] = value["value"]
            elif "dict" in node_type:
                final_dict[key] = {}
            else:
                children = []
                for local_node in local_nodes:
                    children.extend(graph.get_children_by_node_type(local_node, node_type))

                if value["required"] and not children:
                    raise ValueError(f"No child with type {node_type} found")

                child_values = [build_json(child, graph) for child in children]
                final_dict[key] = list(child_values) if value["list"] else (child_values[0] if child_values else None)

        return final_dict

    local_nodes = get_local_nodes(root, graph)
    return recursive_build_json(local_nodes, graph)
