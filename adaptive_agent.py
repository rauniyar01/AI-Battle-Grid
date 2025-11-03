import random
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Set, Tuple

MOVE_VECTORS: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

# Participants edit only this file.
# Implement the decide_move() method to control your agent.
# Valid moves: 'UP', 'DOWN', 'LEFT', 'RIGHT'


class Agent:
    def __init__(self, name: str):
        self.name = name
        self._recent_positions: Deque[Tuple[int, int]] = deque(maxlen=6)
        self._target_food: Optional[Tuple[int, int]] = None

    def decide_move(self, game_state) -> str:
        """
        Decide next move based on the provided game_state (GameState dataclass).
        Accessible fields:
            - game_state.grid_size (int)
            - game_state.you (your agent's name, str)
            - game_state.positions (dict name -> (r,c))
            - game_state.food (set of (r,c))
            - game_state.alive (set of names)
            - game_state.scores (dict name -> int)
            - game_state.turn (int)
            - game_state.max_turns (int)
        Return one of: 'UP', 'DOWN', 'LEFT', 'RIGHT'
        """
        grid = game_state.grid_size
        my_pos = game_state.positions[self.name]
        self._remember_position(my_pos)

        opponents = {
            name: pos
            for name, pos in game_state.positions.items()
            if name != self.name and name in game_state.alive
        }
        opponent_positions: Set[Tuple[int, int]] = set(opponents.values())
        danger_next = self._project_opponent_moves(opponent_positions, grid)

        food_cells = set(game_state.food)
        food_threat = self._food_threat(food_cells, opponent_positions, grid)

        if self._target_food not in food_cells:
            self._target_food = None

        distance_field = self._distance_field(
            start=my_pos,
            forbidden=set(),
            grid_size=grid,
            limit=None,
        )

        if not self._target_food and food_cells:
            self._target_food = self._choose_food_target(
                foods=food_cells,
                distance_field=distance_field,
                food_threat=food_threat,
                grid_size=grid,
            )

        nearest_food = self._nearest_food_by_distance(distance_field, food_cells)

        distance_cache: Dict[Tuple[Tuple[int, int], Optional[Tuple[int, int]]], Optional[int]] = {}

        def dist_from(cell: Tuple[int, int], target: Optional[Tuple[int, int]]) -> Optional[int]:
            key = (cell, target)
            if key not in distance_cache:
                distance_cache[key] = self._shortest_distance(
                    start=cell,
                    target=target,
                    grid_size=grid,
                )
            return distance_cache[key]

        candidate_moves = []
        prev_pos = self._previous_position()

        greedy_move: Optional[str] = None
        if nearest_food is not None:
            greedy_move = self._move_toward(
                start=my_pos,
                target=nearest_food,
                opponent_positions=opponent_positions,
                danger=danger_next,
                grid_size=grid,
            )

        for move, (dr, dc) in MOVE_VECTORS.items():
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if not self._in_bounds(nxt, grid):
                continue

            score = 0.0
            risk = 0.0

            if nxt in opponent_positions:
                risk += 160.0

            if nxt in danger_next:
                penalty = 95.0
                if nxt in food_cells:
                    opp_dist = food_threat.get(nxt, grid * 2)
                    if opp_dist > 0:
                        penalty *= 0.35
                    elif opp_dist == 0:
                        penalty *= 0.6
                risk += penalty

            if prev_pos and nxt == prev_pos:
                risk += 10.0
            if nxt in self._recent_positions:
                risk += 6.0

            future_safe = self._safe_exit_count(
                position=nxt,
                blocked=opponent_positions,
                danger=danger_next,
                grid_size=grid,
            )
            score += future_safe * 12.0

            score += self._distance_from_wall_bonus(nxt, grid) * 1.8

            if nxt in food_cells:
                score += 150.0
            else:
                if self._target_food:
                    dist_to_target = dist_from(nxt, self._target_food)
                    if dist_to_target is not None:
                        score += max(0.0, 70.0 - dist_to_target * 12.0)
                if nearest_food is not None:
                    dist_to_food = dist_from(nxt, nearest_food)
                    if dist_to_food is not None:
                        score += max(0.0, 40.0 - dist_to_food * 10.0)

            adjacency = self._adjacent_opponent_count(nxt, opponent_positions)
            score -= adjacency * 18.0

            total_score = score - risk + random.random() * 1e-4
            candidate_moves.append((total_score, move))

        if not candidate_moves:
            return random.choice(list(MOVE_VECTORS.keys()))

        best_score, best_move = max(candidate_moves, key=lambda item: item[0])

        if greedy_move:
            for candidate_score, move in candidate_moves:
                if move == greedy_move and candidate_score >= best_score - 5.0:
                    best_score, best_move = candidate_score, move
                    break

        return best_move

    # ---------- Helper methods ----------
    @staticmethod
    def _in_bounds(cell: Tuple[int, int], grid_size: int) -> bool:
        r, c = cell
        return 0 <= r < grid_size and 0 <= c < grid_size

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _remember_position(self, position: Tuple[int, int]) -> None:
        if not self._recent_positions or self._recent_positions[-1] != position:
            self._recent_positions.append(position)

    def _previous_position(self) -> Optional[Tuple[int, int]]:
        if len(self._recent_positions) >= 2:
            return self._recent_positions[-2]
        return None

    def _project_opponent_moves(
        self,
        opponent_positions: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Set[Tuple[int, int]]:
        danger: Set[Tuple[int, int]] = set()
        for pos in opponent_positions:
            for dr, dc in MOVE_VECTORS.values():
                nxt = (pos[0] + dr, pos[1] + dc)
                if self._in_bounds(nxt, grid_size):
                    danger.add(nxt)
        return danger

    def _food_threat(
        self,
        foods: Set[Tuple[int, int]],
        opponents: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Dict[Tuple[int, int], int]:
        threat: Dict[Tuple[int, int], int] = {}
        if not foods:
            return threat
        default_dist = grid_size * 3
        for food in foods:
            if opponents:
                threat[food] = min(self._manhattan(op, food) for op in opponents)
            else:
                threat[food] = default_dist
        return threat

    def _choose_food_target(
        self,
        foods: Set[Tuple[int, int]],
        distance_field: Dict[Tuple[int, int], int],
        food_threat: Dict[Tuple[int, int], int],
        grid_size: int,
    ) -> Optional[Tuple[int, int]]:
        best_food: Optional[Tuple[int, int]] = None
        best_score: Optional[float] = None

        for food in foods:
            dist_self = distance_field.get(food)
            if dist_self is None:
                continue
            opp_dist = food_threat.get(food, grid_size * 3)
            margin = opp_dist - dist_self
            center_bias = self._distance_from_wall_bonus(food, grid_size)
            score = margin * 45.0 + center_bias * 3.0 - dist_self * 6.0
            if best_score is None or score > best_score:
                best_score = score
                best_food = food

        return best_food

    def _distance_field(
        self,
        start: Tuple[int, int],
        forbidden: Set[Tuple[int, int]],
        grid_size: int,
        limit: Optional[int],
    ) -> Dict[Tuple[int, int], int]:
        queue: Deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        visited: Set[Tuple[int, int]] = {start}
        field: Dict[Tuple[int, int], int] = {start: 0}

        while queue:
            cell, dist = queue.popleft()
            if limit is not None and dist >= limit:
                continue
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size) or nxt in visited or nxt in forbidden:
                    continue
                visited.add(nxt)
                field[nxt] = dist + 1
                queue.append((nxt, dist + 1))
        return field

    def _shortest_distance(
        self,
        start: Tuple[int, int],
        target: Optional[Tuple[int, int]],
        grid_size: int,
        limit: Optional[int] = None,
    ) -> Optional[int]:
        if target is None:
            return None
        if start == target:
            return 0
        field = self._distance_field(
            start=start,
            forbidden=set(),
            grid_size=grid_size,
            limit=limit,
        )
        return field.get(target)

    def _nearest_food_by_distance(
        self,
        distance_field: Dict[Tuple[int, int], int],
        foods: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        nearest: Optional[Tuple[int, int]] = None
        best_dist: Optional[int] = None
        for food in foods:
            dist = distance_field.get(food)
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                nearest = food
        return nearest

    def _move_toward(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        opponent_positions: Set[Tuple[int, int]],
        danger: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Optional[str]:
        best_move: Optional[str] = None
        best_dist = self._manhattan(start, target)
        fallback_move: Optional[str] = None

        for move, (dr, dc) in MOVE_VECTORS.items():
            nxt = (start[0] + dr, start[1] + dc)
            if not self._in_bounds(nxt, grid_size):
                continue
            if nxt in opponent_positions:
                continue
            dist = self._manhattan(nxt, target)
            if dist < best_dist:
                if nxt not in danger:
                    return move
                if fallback_move is None:
                    fallback_move = move
        return fallback_move

    def _safe_exit_count(
        self,
        position: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        danger: Set[Tuple[int, int]],
        grid_size: int,
    ) -> int:
        exits = 0
        for dr, dc in MOVE_VECTORS.values():
            nxt = (position[0] + dr, position[1] + dc)
            if not self._in_bounds(nxt, grid_size):
                continue
            if nxt in blocked or nxt in danger:
                continue
            exits += 1
        return exits

    def _distance_from_wall_bonus(self, position: Tuple[int, int], grid_size: int) -> float:
        r, c = position
        margin = min(r, c, grid_size - 1 - r, grid_size - 1 - c)
        return float(margin)

    def _adjacent_opponent_count(
        self,
        position: Tuple[int, int],
        opponents: Iterable[Tuple[int, int]],
    ) -> int:
        adj = 0
        for op in opponents:
            if self._manhattan(position, op) == 1:
                adj += 1
        return adj
