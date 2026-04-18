"""LLM and route-graph helpers for high-level planning."""

import json
import os
from pathlib import Path
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


def _validate_graph(graph: dict[str, Any]) -> None:
    """Raise RuntimeError if the graph dict is missing required keys/values."""
    required_keys = {'start_checkpoint', 'checkpoints', 'edges', 'goal_aliases'}
    missing = required_keys - set(graph.keys())
    if missing:
        raise RuntimeError(
            f'Route graph missing required keys: {sorted(missing)}'
        )

    checkpoints = graph['checkpoints']
    if graph['start_checkpoint'] not in checkpoints:
        raise RuntimeError('start_checkpoint must exist in checkpoints')

    for name, checkpoint in checkpoints.items():
        for key in ('x', 'y', 'yaw', 'description'):
            if key not in checkpoint:
                raise RuntimeError(
                    f"Checkpoint '{name}' missing required key '{key}'"
                )

    for edge in graph['edges']:
        from_node = edge.get('from')
        to_node = edge.get('to')
        if from_node not in checkpoints or to_node not in checkpoints:
            raise RuntimeError(f'Edge references unknown checkpoint: {edge}')

    for alias, goals in graph['goal_aliases'].items():
        if not goals:
            raise RuntimeError(f"Goal alias '{alias}' has no targets")
        for goal in goals:
            if goal not in checkpoints:
                raise RuntimeError(
                    f"Goal alias '{alias}' references unknown checkpoint '{goal}'"
                )


def load_route_graph_from_map_poses(
    map_poses_path: str, map_name: str
) -> tuple[dict[str, Any], str]:
    """Load a route graph and sidecar path from a unified ``map_poses.yaml``.

    Parameters
    ----------
    map_poses_path:
        Absolute (or workspace-relative) path to ``map_poses.yaml``.
    map_name:
        The PGM filename key inside ``maps:``, e.g. ``"warehouse.pgm"``.
        If the caller omits the ``.pgm`` suffix it is added automatically.

    Returns
    -------
    (graph, sidecar_path)
        ``graph`` is a validated route-graph dict ready for :func:`plan_route`.
        ``sidecar_path`` is the string value of the ``sidecar`` field (may be
        empty/null — callers treat it as ``""`` when absent).
    """
    import yaml  # available in every ROS 2 Humble Python env

    if not map_name.endswith('.pgm'):
        map_name = map_name + '.pgm'

    p = Path(map_poses_path).expanduser()
    if not p.is_file():
        # Try resolving relative to the workspace root (next to install/).
        ws = Path(__file__).resolve().parents[4]
        p = ws / map_poses_path
    if not p.is_file():
        raise RuntimeError(f'map_poses file not found: {map_poses_path}')

    with p.open() as fh:
        data = yaml.safe_load(fh)

    maps = data.get('maps') or {}
    if map_name not in maps:
        raise RuntimeError(
            f"map_poses.yaml has no entry for '{map_name}'. "
            f"Available maps: {sorted(maps.keys())}"
        )

    entry = maps[map_name] or {}
    raw_graph = entry.get('route_graph')
    if not raw_graph:
        raise RuntimeError(
            f"map_poses.yaml entry for '{map_name}' has no route_graph block. "
            "Add checkpoints, edges, and goal_aliases under route_graph."
        )

    graph: dict[str, Any] = dict(raw_graph)
    if not graph.get('edges'):
        graph['edges'] = []
    if not graph.get('goal_aliases'):
        graph['goal_aliases'] = {}

    _validate_graph(graph)

    sidecar_path: str = entry.get('sidecar') or ''
    return graph, sidecar_path


def plan_route(
    *,
    goal_request: str,
    blocked_edges: set[tuple[str, str]],
    failure_reason: str,
    current_checkpoint: str,
    goal_aliases: dict[str, list[str]],
    checkpoints: dict[str, dict[str, Any]],
    edges: set[tuple[str, str]],
    planner_notes: str,
    max_attempts: int,
    publish_status: Callable[[str], None] | None = None,
) -> tuple[str, list[str]]:
    """Ask the LLM for a legal route, retrying invalid decisions."""
    last_error = 'No route decision attempts were made'

    for attempt in range(1, max_attempts + 1):
        decision = make_decision(
            goal_request=goal_request,
            blocked_edges=blocked_edges,
            failure_reason=failure_reason,
            current_checkpoint=current_checkpoint,
            goal_aliases=goal_aliases,
            checkpoints=checkpoints,
            edges=edges,
            planner_notes=planner_notes,
        )
        try:
            route = validate_decision(
                decision=decision,
                blocked_edges=blocked_edges,
                current_checkpoint=current_checkpoint,
                goal_aliases=goal_aliases,
                checkpoints=checkpoints,
                edges=edges,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if publish_status is not None:
                publish_status(
                    f'Ignoring invalid route from LLM on attempt {attempt}/'
                    f'{max_attempts}: {last_error}'
                )
            continue

        return decision['goal_alias'], route

    raise RuntimeError(
        'LLM did not return a valid legal route after '
        f'{max_attempts} attempts: {last_error}'
    )


def make_decision(
    *,
    goal_request: str,
    blocked_edges: set[tuple[str, str]],
    failure_reason: str,
    current_checkpoint: str,
    goal_aliases: dict[str, list[str]],
    checkpoints: dict[str, dict[str, Any]],
    edges: set[tuple[str, str]],
    planner_notes: str,
) -> dict[str, Any]:
    """Ask the LLM to choose a legal route through the checkpoint graph."""
    provider = os.environ.get('LLM_PROVIDER', '')
    model = os.environ.get('LLM_MODEL', '')

    if not provider:
        raise RuntimeError(
            'Missing LLM_PROVIDER environment variable. '
            'Set it in your .env file (e.g. LLM_PROVIDER=openai).'
        )
    if not model:
        raise RuntimeError(
            'Missing LLM_MODEL environment variable. '
            'Set it in your .env file (e.g. LLM_MODEL=gpt-4o).'
        )

    try:
        llm = init_chat_model(
            model=model,
            model_provider=provider,
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(
            f'Failed to initialise LLM ({provider}/{model}): {exc}'
        ) from exc

    system_content = (
        'You are the high-level route decision layer for a TurtleBot. '
        'Do not create arbitrary coordinates. '
        'Choose only from the checkpoint graph provided in the user JSON. '
        'Respond with valid JSON only — no markdown, no explanation, no '
        'code fences. Use exactly this schema: '
        '{"goal_alias": string, "route": [string, ...], "reason": string}. '
        'The route must begin at current_checkpoint, end at a checkpoint '
        'allowed by the chosen goal_alias, and only use allowed edges that '
        f'are not blocked. {planner_notes}'
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(
            content=json.dumps(
                build_decision_context(
                    goal_request=goal_request,
                    blocked_edges=blocked_edges,
                    failure_reason=failure_reason,
                    current_checkpoint=current_checkpoint,
                    goal_aliases=goal_aliases,
                    checkpoints=checkpoints,
                    edges=edges,
                )
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        raise RuntimeError(f'LLM request failed: {exc}') from exc

    content = response.content
    # Strip markdown code fences if a model wraps its output anyway.
    content = content.strip()
    if content.startswith('```'):
        content = content.split('```')[1]
        if content.startswith('json'):
            content = content[4:]

    try:
        decision = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'LLM returned non-JSON response: {exc}'
        ) from exc

    if 'goal_alias' not in decision or 'route' not in decision:
        raise RuntimeError(
            'LLM response must include goal_alias and route fields'
        )

    if not isinstance(decision['route'], list):
        raise RuntimeError('LLM route must be a JSON array')

    decision.setdefault('reason', '')
    return decision


def validate_decision(
    *,
    decision: dict[str, Any],
    blocked_edges: set[tuple[str, str]],
    current_checkpoint: str,
    goal_aliases: dict[str, list[str]],
    checkpoints: dict[str, dict[str, Any]],
    edges: set[tuple[str, str]],
) -> list[str]:
    """Validate that the LLM decision stays within the allowed graph."""
    goal_alias = decision['goal_alias']
    if goal_alias not in goal_aliases:
        raise RuntimeError(f"Unknown goal alias '{goal_alias}'")

    route = [str(node) for node in decision['route']]
    if len(route) < 2:
        raise RuntimeError('Route must include at least start and goal')

    if route[0] != current_checkpoint:
        raise RuntimeError(
            'Route must start at the current checkpoint '
            f"'{current_checkpoint}'"
        )

    allowed_goals = set(goal_aliases[goal_alias])
    if route[-1] not in allowed_goals:
        raise RuntimeError(
            f"Route for goal alias '{goal_alias}' must end at one of "
            f'{sorted(allowed_goals)}'
        )

    for node in route:
        if node not in checkpoints:
            raise RuntimeError(f"Unknown checkpoint '{node}' in route")

    for from_node, to_node in zip(route, route[1:]):
        if (from_node, to_node) not in edges:
            raise RuntimeError(
                f"Route includes invalid edge '{from_node}->{to_node}'"
            )
        if (from_node, to_node) in blocked_edges:
            raise RuntimeError(
                f"Route reuses blocked edge '{from_node}->{to_node}'"
            )

    return route


def build_decision_context(
    *,
    goal_request: str,
    blocked_edges: set[tuple[str, str]],
    failure_reason: str,
    current_checkpoint: str,
    goal_aliases: dict[str, list[str]],
    checkpoints: dict[str, dict[str, Any]],
    edges: set[tuple[str, str]],
) -> dict[str, Any]:
    """Assemble the structured graph context for route selection."""
    return {
        'goal_request': goal_request,
        'current_checkpoint': current_checkpoint,
        'goal_aliases': goal_aliases,
        'checkpoint_descriptions': {
            name: checkpoint['description']
            for name, checkpoint in checkpoints.items()
        },
        'allowed_edges': [
            {'from': from_node, 'to': to_node}
            for from_node, to_node in sorted(edges)
            if (from_node, to_node) not in blocked_edges
        ],
        'blocked_edges': [
            {'from': from_node, 'to': to_node}
            for from_node, to_node in sorted(blocked_edges)
        ],
        'last_failure_reason': failure_reason,
    }
