"""LLM and route-graph helpers for high-level planning."""

import json
from pathlib import Path
from typing import Any, Callable
from urllib import error, request


def load_route_graph(graph_path: str) -> dict[str, Any]:
    """Load and validate the route graph JSON from disk."""
    if not graph_path:
        raise RuntimeError('route_graph_path parameter is required')

    path = Path(graph_path)
    if not path.is_file():
        raise RuntimeError(f'Route graph file not found: {graph_path}')

    with path.open('r', encoding='utf-8') as handle:
        graph = json.load(handle)

    required_keys = {
        'start_checkpoint',
        'checkpoints',
        'edges',
        'goal_aliases',
    }
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
                    f"Goal alias '{alias}' references unknown checkpoint "
                    f"'{goal}'"
                )

    return graph


def plan_route(
    *,
    goal_request: str,
    blocked_edges: set[tuple[str, str]],
    failure_reason: str,
    current_checkpoint: str,
    goal_aliases: dict[str, list[str]],
    checkpoints: dict[str, dict[str, Any]],
    edges: set[tuple[str, str]],
    api_key: str,
    model: str,
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
            api_key=api_key,
            model=model,
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
                    f'Ignoring invalid route from Groq on attempt {attempt}/'
                    f'{max_attempts}: {last_error}'
                )
            continue

        return decision['goal_alias'], route

    raise RuntimeError(
        'Groq did not return a valid legal route after '
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
    api_key: str,
    model: str,
    planner_notes: str,
) -> dict[str, Any]:
    """Ask the LLM to choose a legal route through the checkpoint graph."""
    if not api_key:
        raise RuntimeError(
            'Missing Groq API key. Set the groq_api_key parameter or '
            'GROQ_API_KEY environment variable.'
        )

    payload = {
        'model': model,
        'temperature': 0.2,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You are the high-level route decision layer for a '
                    'TurtleBot. '
                    'Do not create arbitrary coordinates. '
                    'Choose only from the checkpoint graph provided in '
                    'the user JSON. '
                    'Output JSON only with this schema: '
                    '{"goal_alias": string, "route": [string, ...], '
                    '"reason": string}. '
                    'The route must begin at current_checkpoint, end at a '
                    'checkpoint allowed by the chosen goal_alias, and '
                    'only use allowed edges that are not blocked. '
                    f'{planner_notes}'
                ),
            },
            {
                'role': 'user',
                'content': json.dumps(
                    build_decision_context(
                        goal_request=goal_request,
                        blocked_edges=blocked_edges,
                        failure_reason=failure_reason,
                        current_checkpoint=current_checkpoint,
                        goal_aliases=goal_aliases,
                        checkpoints=checkpoints,
                        edges=edges,
                    )
                ),
            },
        ],
        'response_format': {'type': 'json_object'},
    }

    body = json.dumps(payload).encode('utf-8')
    http_request = request.Request(
        'https://api.groq.com/openai/v1/chat/completions',
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'nav2-llm-demo/0.1 (+groq api client)',
        },
        method='POST',
    )

    try:
        with request.urlopen(http_request, timeout=30) as response:
            response_body = response.read().decode('utf-8')
    except error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(
            f'Groq API returned HTTP {exc.code}: {details}'
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f'Could not reach Groq API: {exc.reason}') from exc

    response_json = json.loads(response_body)
    content = response_json['choices'][0]['message']['content']
    decision = json.loads(content)

    if 'goal_alias' not in decision or 'route' not in decision:
        raise RuntimeError(
            'Groq response must include goal_alias and route fields'
        )

    if not isinstance(decision['route'], list):
        raise RuntimeError('Groq route must be a JSON array')

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
