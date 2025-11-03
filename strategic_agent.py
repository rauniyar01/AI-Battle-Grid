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

        # Track alive opponents only; dead agents are irrelevant blockers.
        opponents = {
            name: pos
            for name, pos in game_state.positions.items()
            if name != self.name and name in game_state.alive
        }
        opponent_positions: Set[Tuple[int, int]] = set(opponents.values())

        # Cells opponents can reach next turn are high-risk.
        danger_next: Set[Tuple[int, int]] = set()
        for pos in opponent_positions:
            for dr, dc in MOVE_VECTORS.values():
                nxt = (pos[0] + dr, pos[1] + dc)
                if self._in_bounds(nxt, grid):
                    danger_next.add(nxt)

        # Maintain a moving target so we keep pursuing promising food.
        food_cells = set(game_state.food)
        if self._target_food not in food_cells:
            self._target_food = None

        if not self._target_food and food_cells:
            self._target_food = self._choose_food_target(
                start=my_pos,
                foods=food_cells,
                opponents=opponent_positions,
                blocked=opponent_positions,
                grid_size=grid,
            )

        candidate_moves = []
        prev_pos = self._previous_position()

        for move, (dr, dc) in MOVE_VECTORS.items():
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if not self._in_bounds(nxt, grid):
                continue

            score = 0.0
            risk = 0.0

            if nxt in opponent_positions:
                risk += 150.0  # likely swap or head-on collision
            if nxt in danger_next:
                risk += 110.0

            if prev_pos and nxt == prev_pos:
                risk += 18.0  # discourage undoing last step unless desperate
            if nxt in self._recent_positions:
                risk += 10.0

            # Prefer spots that still offer future escape options.
            future_safe = self._safe_exit_count(
                position=nxt,
                blocked=opponent_positions,
                danger=danger_next,
                grid_size=grid,
            )
            score += future_safe * 11.0

            # Staying nearer to the board center offers more room to maneuver.
            score += self._distance_from_wall_bonus(nxt, grid) * 2.5

            # Reward food and paths toward the current target.
            if nxt in food_cells:
                score += 180.0
            elif self._target_food:
                dist_to_target = self._shortest_distance(
                    start=nxt,
                    target=self._target_food,
                    blocked=opponent_positions,
                    grid_size=grid,
                )
                if dist_to_target is not None:
                    score += max(0.0, 60.0 - dist_to_target * 9.0)
                else:
                    risk += 35.0  # moving away from the only reachable food
            else:
                dist_any_food = self._nearest_food_distance(
                    start=nxt,
                    foods=food_cells,
                    blocked=opponent_positions,
                    grid_size=grid,
                )
                if dist_any_food is not None:
                    score += max(0.0, 30.0 - dist_any_food * 6.0)

            # Mild bonus for keeping distance from immediate opponent adjacency.
            adjacency_penalty = self._adjacent_opponent_count(nxt, opponent_positions)
            score -= adjacency_penalty * 25.0

            # Combine risk and a tiny random jitter for deterministic tie-breaking.
            total_score = score - risk + random.random() * 1e-4
            candidate_moves.append((total_score, move))

        if not candidate_moves:
            return random.choice(list(MOVE_VECTORS.keys()))

        best_move = max(candidate_moves, key=lambda item: item[0])[1]
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

    def _choose_food_target(
        self,
        start: Tuple[int, int],
        foods: Set[Tuple[int, int]],
        opponents: Set[Tuple[int, int]],
        blocked: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Optional[Tuple[int, int]]:
        best_food: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[float, float, int, float]] = None

        for food in foods:
            dist_self = self._shortest_distance(start, food, blocked, grid_size)
            if dist_self is None:
                continue

            if opponents:
                opp_best = min(self._manhattan(op, food) for op in opponents)
            else:
                opp_best = grid_size * 4

            margin = opp_best - dist_self
            crowd = sum(1 for op in opponents if self._manhattan(op, food) <= 2)
            center_bias = self._distance_from_wall_bonus(food, grid_size)

            key = (-margin, dist_self, crowd, -center_bias)
            if best_key is None or key < best_key:
                best_key = key
                best_food = food

        return best_food

    def _shortest_distance(
        self,
        start: Tuple[int, int],
        target: Optional[Tuple[int, int]],
        blocked: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Optional[int]:
        if target is None:
            return None
        if start == target:
            return 0

        queue: Deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        visited: Set[Tuple[int, int]] = {start}

        while queue:
            cell, dist = queue.popleft()
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size) or nxt in visited or nxt in blocked:
                    continue
                if nxt == target:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def _nearest_food_distance(
        self,
        start: Tuple[int, int],
        foods: Set[Tuple[int, int]],
        blocked: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Optional[int]:
        if not foods:
            return None

        queue: Deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        visited: Set[Tuple[int, int]] = {start}

        while queue:
            cell, dist = queue.popleft()
            if cell in foods:
                return dist
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size) or nxt in visited or nxt in blocked:
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

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
