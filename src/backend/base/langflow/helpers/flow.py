from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from fastapi import HTTPException
from lfx.log.logger import logger
from pydantic.v1 import BaseModel, Field, create_model
from sqlmodel import select

from langflow.schema.schema import INPUT_FIELD_NAME
from langflow.services.database.models.flow.model import Flow, FlowRead
from langflow.services.deps import get_settings_service, session_scope

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from lfx.graph.graph.base import Graph
    from lfx.graph.schema import RunOutputs
    from lfx.graph.vertex.base import Vertex

    from langflow.schema.data import Data

INPUT_TYPE_MAP = {
    "ChatInput": {"type_hint": "Optional[str]", "default": '""'},
    "TextInput": {"type_hint": "Optional[str]", "default": '""'},
    "JSONInput": {"type_hint": "Optional[dict]", "default": "{}"},
}


async def list_flows(*, user_id: str | None = None) -> list[Data]:
    if not user_id:
        msg = "Session is invalid"
        raise ValueError(msg)
    try:
        async with session_scope() as session:
            uuid_user_id = UUID(user_id) if isinstance(user_id, str) else user_id
            stmt = select(Flow).where(Flow.user_id == uuid_user_id).where(Flow.is_component == False)  # noqa: E712
            flows = (await session.exec(stmt)).all()

            return [flow.to_data() for flow in flows]
    except Exception as e:
        msg = f"Error listing flows: {e}"
        raise ValueError(msg) from e


async def load_flow(
    user_id: str, flow_id: str | None = None, flow_name: str | None = None, tweaks: dict | None = None
) -> Graph:
    from lfx.graph.graph.base import Graph

    from langflow.processing.process import process_tweaks

    if not flow_id and not flow_name:
        msg = "Flow ID or Flow Name is required"
        raise ValueError(msg)
    if not flow_id and flow_name:
        flow_id = await find_flow(flow_name, user_id)
        if not flow_id:
            msg = f"Flow {flow_name} not found"
            raise ValueError(msg)

    async with session_scope() as session:
        graph_data = flow.data if (flow := await session.get(Flow, flow_id)) else None
    if not graph_data:
        msg = f"Flow {flow_id} not found"
        raise ValueError(msg)
    if tweaks:
        graph_data = process_tweaks(graph_data=graph_data, tweaks=tweaks)
    return Graph.from_payload(graph_data, flow_id=flow_id, user_id=user_id)


async def find_flow(flow_name: str, user_id: str) -> str | None:
    async with session_scope() as session:
        uuid_user_id = UUID(user_id) if isinstance(user_id, str) else user_id
        stmt = select(Flow).where(Flow.name == flow_name).where(Flow.user_id == uuid_user_id)
        flow = (await session.exec(stmt)).first()
        return flow.id if flow else None


async def run_flow(
    inputs: dict | list[dict] | None = None,
    tweaks: dict | None = None,
    flow_id: str | None = None,
    flow_name: str | None = None,
    output_type: str | None = "chat",
    user_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    graph: Graph | None = None,
) -> list[RunOutputs]:
    if user_id is None:
        msg = "Session is invalid"
        raise ValueError(msg)
    if graph is None:
        graph = await load_flow(user_id, flow_id, flow_name, tweaks)
    if run_id:
        graph.set_run_id(UUID(run_id))
    if session_id:
        graph.session_id = session_id
    if user_id:
        graph.user_id = user_id

    if inputs is None:
        inputs = []
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs_list = []
    inputs_components = []
    types = []
    for input_dict in inputs:
        inputs_list.append({INPUT_FIELD_NAME: cast("str", input_dict.get("input_value"))})
        inputs_components.append(input_dict.get("components", []))
        types.append(input_dict.get("type", "chat"))

    outputs = [
        vertex.id
        for vertex in graph.vertices
        if output_type == "debug"
        or (
            vertex.is_output and (output_type == "any" or output_type in vertex.id.lower())  # type: ignore[operator]
        )
    ]

    fallback_to_env_vars = get_settings_service().settings.fallback_to_env_var

    return await graph.arun(
        inputs_list,
        outputs=outputs,
        inputs_components=inputs_components,
        types=types,
        fallback_to_env_vars=fallback_to_env_vars,
    )


def generate_function_for_flow(
    inputs: list[Vertex], flow_id: str, user_id: str | UUID | None
) -> Callable[..., Awaitable[Any]]:
    """Generate a dynamic flow function based on the given inputs and flow ID.

    Args:
        inputs (List[Vertex]): The list of input vertices for the flow.
        flow_id (str): The ID of the flow.
        user_id (str | UUID | None): The user ID associated with the flow.

    Returns:
        Coroutine: The dynamic flow function.

    Raises:
        None

    Example:
        inputs = [vertex1, vertex2]
        flow_id = "my_flow"
        function = generate_function_for_flow(inputs, flow_id)
        result = function(input1, input2)
    """
    # Prepare function arguments with type hints and default values
    # This computation is reasonably optimal for a small list and cannot be deferred to a helper for string rendering
    args = []
    original_arg_names = []
    arg_var_names = []
    for input_ in inputs:
        lower_name = input_.display_name.lower().replace(" ", "_")
        type_hint = INPUT_TYPE_MAP[input_.base_name]["type_hint"]
        default = INPUT_TYPE_MAP[input_.base_name]["default"]
        # Avoid building the same string twice for each part
        args.append(f"{lower_name}: {type_hint} = {default}")
        original_arg_names.append(input_.display_name)
        arg_var_names.append(lower_name)

    func_args = ", ".join(args)
    # Precompute zipped lists, avoiding list comprehensions inside join for some performance in large cases
    arg_mappings_pairs = []
    for original_name, var_name in zip(original_arg_names, arg_var_names, strict=True):
        arg_mappings_pairs.append(f'"{original_name}": {var_name}')
    arg_mappings = ", ".join(arg_mappings_pairs)

    # Pre-render the body string, prepare reusable blocks outside the triple quote if possible
    func_body = (
        "from typing import Optional\n"
        f"async def flow_function({func_args}):\n"
        f"    tweaks = {{ {arg_mappings} }}\n"
        "    from langflow.helpers.flow import run_flow\n"
        "    from langchain_core.tools import ToolException\n"
        "    from lfx.base.flow_processing.utils import build_data_from_result_data, format_flow_output_data\n"
        "    try:\n"
        "        run_outputs = await run_flow(\n"
        "            tweaks={key: {'input_value': value} for key, value in tweaks.items()},\n"
        f'            flow_id="{flow_id}",\n'
        f'            user_id="{user_id}"\n'
        "        )\n"
        "        if not run_outputs:\n"
        "            return []\n"
        "        run_output = run_outputs[0]\n"
        "        data = []\n"
        "        if run_output is not None:\n"
        "            for output in run_output.outputs:\n"
        "                if output:\n"
        "                    data.extend(build_data_from_result_data(output))\n"
        "        return format_flow_output_data(data)\n"
        "    except Exception as e:\n"
        "        raise ToolException(f'Error running flow: ' + e)\n"
    )

    # Try to reduce time spent in compile/exec by minimizing scope size
    compiled_func = compile(func_body, "<string>", "exec")
    # Avoid excessive growth of the globals dictionary by isolating the local scope
    local_scope: dict[str, Any] = {}
    exec(compiled_func, globals(), local_scope)  # noqa: S102
    return local_scope["flow_function"]


def build_function_and_schema(
    flow_data: Data, graph: Graph, user_id: str | UUID | None
) -> tuple[Callable[..., Awaitable[Any]], type[BaseModel]]:
    """Builds a dynamic function and schema for a given flow.

    Args:
        flow_data (Data): The flow record containing information about the flow.
        graph (Graph): The graph representing the flow.
        user_id (str): The user ID associated with the flow.

    Returns:
        Tuple[Callable, BaseModel]: A tuple containing the dynamic function and the schema.
    """
    flow_id = flow_data.id
    inputs = get_flow_inputs(graph)
    dynamic_flow_function = generate_function_for_flow(inputs, flow_id, user_id=user_id)
    # Use a cache to improve performance of schema creation if same schema for same inputs is requested repeatedly
    schema = _cached_schema_builder(flow_data.name, tuple((inp.display_name, inp.description) for inp in inputs))
    return dynamic_flow_function, schema


def get_flow_inputs(graph: Graph) -> list[Vertex]:
    """Retrieves the flow inputs from the given graph.

    Args:
        graph (Graph): The graph object representing the flow.

    Returns:
        List[Data]: A list of input data, where each record contains the ID, name, and description of the input vertex.
    """
    # List comprehension likely optimal for this use-case
    return [vertex for vertex in graph.vertices if vertex.is_input]


def build_schema_from_inputs(name: str, inputs: list[Vertex]) -> type[BaseModel]:
    """Builds a schema from the given inputs.

    Args:
        name (str): The name of the schema.
        inputs (List[tuple[str, str, str]]): A list of tuples representing the inputs.
            Each tuple contains three elements: the input name, the input type, and the input description.

    Returns:
        BaseModel: The schema model.

    """
    fields = {}
    for input_ in inputs:
        field_name = input_.display_name.lower().replace(" ", "_")
        description = input_.description
        fields[field_name] = (str, Field(default="", description=description))
    return create_model(name, **fields)


def get_arg_names(inputs: list[Vertex]) -> list[dict[str, str]]:
    """Returns a list of dictionaries containing the component name and its corresponding argument name.

    Args:
        inputs (List[Vertex]): A list of Vertex objects representing the inputs.

    Returns:
        List[dict[str, str]]: A list of dictionaries, where each dictionary contains the component name and its
            argument name.
    """
    return [
        {"component_name": input_.display_name, "arg_name": input_.display_name.lower().replace(" ", "_")}
        for input_ in inputs
    ]


async def get_flow_by_id_or_endpoint_name(flow_id_or_name: str, user_id: str | UUID | None = None) -> FlowRead | None:
    async with session_scope() as session:
        endpoint_name = None
        try:
            flow_id = UUID(flow_id_or_name)
            flow = await session.get(Flow, flow_id)
        except ValueError:
            endpoint_name = flow_id_or_name
            stmt = select(Flow).where(Flow.endpoint_name == endpoint_name)
            if user_id:
                uuid_user_id = UUID(user_id) if isinstance(user_id, str) else user_id
                stmt = stmt.where(Flow.user_id == uuid_user_id)
            flow = (await session.exec(stmt)).first()
        if flow is None:
            raise HTTPException(status_code=404, detail=f"Flow identifier {flow_id_or_name} not found")
        return FlowRead.model_validate(flow, from_attributes=True)


async def generate_unique_flow_name(flow_name, user_id, session):
    original_name = flow_name
    n = 1
    while True:
        # Check if a flow with the given name exists
        existing_flow = (
            await session.exec(
                select(Flow).where(
                    Flow.name == flow_name,
                    Flow.user_id == user_id,
                )
            )
        ).first()

        # If no flow with the given name exists, return the name
        if not existing_flow:
            return flow_name

        # If a flow with the name already exists, append (n) to the name and increment n
        flow_name = f"{original_name} ({n})"
        n += 1


def json_schema_from_flow(flow: Flow) -> dict:
    """Generate JSON schema from flow input nodes."""
    from lfx.graph.graph.base import Graph

    # Get the flow's data which contains the nodes and their configurations
    flow_data = flow.data or {}

    graph = Graph.from_payload(flow_data)
    input_nodes = [vertex for vertex in graph.vertices if vertex.is_input]

    properties = {}
    required = []
    for node in input_nodes:
        node_data = node.data["node"]
        template = node_data["template"]

        for field_name, field_data in template.items():
            if field_data != "Component" and field_data.get("show", False) and not field_data.get("advanced", False):
                field_type = field_data.get("type", "string")
                properties[field_name] = {
                    "type": field_type,
                    "description": field_data.get("info", f"Input for {field_name}"),
                }
                # Update field_type in properties after determining the JSON Schema type
                if field_type == "str":
                    field_type = "string"
                elif field_type == "int":
                    field_type = "integer"
                elif field_type == "float":
                    field_type = "number"
                elif field_type == "bool":
                    field_type = "boolean"
                else:
                    logger.warning(f"Unknown field type: {field_type} defaulting to string")
                    field_type = "string"
                properties[field_name]["type"] = field_type

                if field_data.get("required", False):
                    required.append(field_name)

    return {"type": "object", "properties": properties, "required": required}


@lru_cache(maxsize=64)
def _cached_schema_builder(name: str, inputs: tuple) -> type[BaseModel]:
    # Internal helper for performance: build (and cache) the schema model for an input signature
    # Inputs argument is tuple of (display_name, description) for each Vertex (name must remain a string)
    fields = {}
    for inp in inputs:
        field_name = inp[0].lower().replace(" ", "_")
        description = inp[1]
        fields[field_name] = (str, Field(default="", description=description))
    return create_model(name, **fields)
