import heapq
import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

MOVE_VECTORS: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

MOVE_NAMES: List[str] = list(MOVE_VECTORS.keys())


class Agent:
    def __init__(self, name: str):
        self.name = name
        self._recent_positions: deque[Tuple[int, int]] = deque(maxlen=6)
        self._last_target: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def decide_move(self, game_state) -> str:
        grid = game_state.grid_size
        my_pos = game_state.positions[self.name]
        self._remember_position(my_pos)

        opponents = {
            n: pos
            for n, pos in game_state.positions.items()
            if n != self.name and n in game_state.alive
        }
        opponent_positions: Set[Tuple[int, int]] = set(opponents.values())
        danger_cells = self._project_danger(opponent_positions, grid)
        food_cells: Set[Tuple[int, int]] = set(game_state.food)

        if self._last_target not in food_cells:
            self._last_target = None

        target, path = self._select_target_via_astar(
            start=my_pos,
            food_cells=food_cells,
            blocked=opponent_positions,
            danger=danger_cells,
            grid_size=grid,
            opponent_positions=opponent_positions,
        )

        if path and len(path) >= 2:
            self._last_target = target
            return self._direction_from_step(path[0], path[1])

        fallback = self._best_immediate_move(
            position=my_pos,
            blocked=opponent_positions,
            danger=danger_cells,
            grid_size=grid,
            food_cells=food_cells,
        )
        if fallback:
            self._last_target = None
            return fallback

        # No evaluated option available -> random legal move to keep playing.
        return self._random_legal_move(my_pos, grid)

    # ------------------------------------------------------------------ #
    # Target selection
    # ------------------------------------------------------------------ #
    def _select_target_via_astar(
        self,
        start: Tuple[int, int],
        food_cells: Set[Tuple[int, int]],
        blocked: Set[Tuple[int, int]],
        danger: Set[Tuple[int, int]],
        grid_size: int,
        opponent_positions: Set[Tuple[int, int]],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
        candidates: List[Tuple[float, Tuple[int, int], List[Tuple[int, int]]]] = []

        for food in food_cells:
            if food in blocked:
                continue  # opponent currently sitting on it -> too risky
            path, path_cost = self._a_star_path(
                start=start,
                goal=food,
                blocked=blocked,
                danger=danger,
                grid_size=grid_size,
                opponent_positions=opponent_positions,
            )
            if not path:
                continue

            dist_self = len(path) - 1
            risk = max(0.0, path_cost - float(dist_self))
            dist_opp = self._best_opponent_distance(
                target=food,
                opponents=opponent_positions,
                blocked=blocked,
                grid_size=grid_size,
                self_position=start,
            )
            if dist_opp is None:
                dist_opp = grid_size * 4
            margin = dist_opp - dist_self
            advantage = max(0, margin)
            disadvantage = max(0, -margin)
            exit_bonus = self._safe_exit_count(path[-1], blocked, danger, grid_size)
            center_bonus = self._distance_from_wall(path[-1], grid_size)
            hazard = sum(1 for cell in path[1:] if cell in danger)

            score = (
                320.0
                - dist_self * 32.0
                - risk * 22.0
                + exit_bonus * 16.0
                + center_bonus * 5.0
                - hazard * 15.0
            )
            score += advantage * 34.0
            score -= disadvantage * 60.0
            if margin == 0:
                score -= 40.0
            if food == self._last_target:
                score += 25.0  # stickiness to reduce oscillation

            candidates.append((score, food, path))

        if not candidates:
            control_targets = self._control_targets(
                start=start,
                blocked=blocked,
                danger=danger,
                grid_size=grid_size,
            )
            for target, base_score in control_targets:
                if target in blocked:
                    continue
                path, path_cost = self._a_star_path(
                    start=start,
                    goal=target,
                    blocked=blocked,
                    danger=danger,
                    grid_size=grid_size,
                    opponent_positions=opponent_positions,
                )
                if not path:
                    continue
                dist_self = len(path) - 1
                risk = max(0.0, path_cost - float(dist_self))
                exit_bonus = self._safe_exit_count(path[-1], blocked, danger, grid_size)
                center_bonus = self._distance_from_wall(path[-1], grid_size)
                hazard = sum(1 for cell in path[1:] if cell in danger)
                score = (
                    base_score * 22.0
                    - dist_self * 18.0
                    - risk * 14.0
                    + exit_bonus * 13.0
                    + center_bonus * 5.5
                    - hazard * 13.0
                )
                candidates.append((score, target, path))

        if not candidates:
            return None, None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_target, best_path = candidates[0]
        if best_score == float("-inf"):
            return None, None
        return best_target, best_path

    # ------------------------------------------------------------------ #
    # Immediate fallback move
    # ------------------------------------------------------------------ #
    def _best_immediate_move(
        self,
        position: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        danger: Set[Tuple[int, int]],
        grid_size: int,
        food_cells: Set[Tuple[int, int]],
    ) -> Optional[str]:
        candidates: List[Tuple[float, str, bool]] = []
        for move, (dr, dc) in MOVE_VECTORS.items():
            nxt = (position[0] + dr, position[1] + dc)
            if not self._in_bounds(nxt, grid_size):
                continue
            if nxt in blocked:
                continue

            score = 0.0
            if nxt in food_cells:
                score += 110.0

            is_risky = nxt in danger
            if is_risky:
                score -= 120.0

            exit_bonus = self._safe_exit_count(nxt, blocked, danger, grid_size)
            score += exit_bonus * 20.0

            center_bonus = self._distance_from_wall(nxt, grid_size)
            score += center_bonus * 6.0

            space = self._reachable_space(nxt, blocked, grid_size, limit=40)
            score += space * 0.35

            if nxt in self._recent_positions:
                score -= 10.0

            proximity = self._nearest_opponent_distance(nxt, blocked)
            if proximity == 0:
                score -= 150.0
            elif proximity == 1:
                score -= 55.0
            elif proximity == 2:
                score -= 18.0

            candidates.append((score, move, is_risky))

        if not candidates:
            return None

        safe_candidates = [c for c in candidates if not c[2]]
        pool = safe_candidates if safe_candidates else candidates
        pool.sort(key=lambda item: item[0], reverse=True)
        top_score = pool[0][0]
        top_moves = [item[1] for item in pool if item[0] == top_score]
        return random.choice(top_moves)

    # ------------------------------------------------------------------ #
    # A* Search with risk-aware costs
    # ------------------------------------------------------------------ #
    def _a_star_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        danger: Set[Tuple[int, int]],
        grid_size: int,
        opponent_positions: Set[Tuple[int, int]],
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        if start == goal:
            return [start], 0.0

        frontier: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(frontier, (self._heuristic(start, goal), 0.0, start))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)

            if current == goal:
                break

            for move in MOVE_NAMES:
                dr, dc = MOVE_VECTORS[move]
                nxt = (current[0] + dr, current[1] + dc)
                if not self._in_bounds(nxt, grid_size):
                    continue
                if nxt != goal and nxt in blocked:
                    continue

                step_cost = 1.0
                if nxt in danger:
                    step_cost += 30.0
                dist_enemy = self._nearest_opponent_distance(nxt, opponent_positions)
                if dist_enemy == 0:
                    step_cost += 120.0
                elif dist_enemy == 1:
                    step_cost += 24.0
                elif dist_enemy == 2:
                    step_cost += 7.0

                wall_penalty = max(0.0, 2.5 - self._distance_from_wall(nxt, grid_size))
                step_cost += wall_penalty

                if nxt in self._recent_positions:
                    step_cost += 3.0

                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(nxt, float("inf")):
                    continue
                cost_so_far[nxt] = new_cost
                priority = new_cost + self._heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, new_cost, nxt))
                came_from[nxt] = current

        if goal not in came_from:
            return None, float("inf")

        path = self._reconstruct_path(came_from, goal)
        return path, cost_so_far[goal]

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def _remember_position(self, position: Tuple[int, int]) -> None:
        if not self._recent_positions or self._recent_positions[-1] != position:
            self._recent_positions.append(position)

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        current: Optional[Tuple[int, int]] = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    @staticmethod
    def _in_bounds(cell: Tuple[int, int], grid_size: int) -> bool:
        r, c = cell
        return 0 <= r < grid_size and 0 <= c < grid_size

    @staticmethod
    def _distance_from_wall(cell: Tuple[int, int], grid_size: int) -> float:
        r, c = cell
        return float(min(r, c, grid_size - 1 - r, grid_size - 1 - c))

    def _shortest_distance(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Optional[int]:
        if start == target:
            return 0
        queue: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            cell, dist = queue.popleft()
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size):
                    continue
                if nxt != target and nxt in blocked:
                    continue
                if nxt in visited:
                    continue
                if nxt == target:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def _best_opponent_distance(
        self,
        target: Tuple[int, int],
        opponents: Set[Tuple[int, int]],
        blocked: Set[Tuple[int, int]],
        grid_size: int,
        self_position: Tuple[int, int],
    ) -> Optional[int]:
        best: Optional[int] = None
        for op in opponents:
            others = set(opponents)
            others.discard(op)
            others.add(self_position)
            neighbor_block = self._project_danger(others, grid_size)
            adjusted_blocked = others | neighbor_block
            adjusted_blocked.discard(target)
            dist = self._shortest_distance(op, target, adjusted_blocked, grid_size)
            if dist is None:
                continue
            if best is None or dist < best:
                best = dist
        return best

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _direction_from_step(
        origin: Tuple[int, int], nxt: Tuple[int, int]
    ) -> str:
        dr = nxt[0] - origin[0]
        dc = nxt[1] - origin[1]
        for move, vec in MOVE_VECTORS.items():
            if vec == (dr, dc):
                return move
        return random.choice(MOVE_NAMES)

    @staticmethod
    def _project_danger(positions: Set[Tuple[int, int]], grid_size: int) -> Set[Tuple[int, int]]:
        danger: Set[Tuple[int, int]] = set()
        for r, c in positions:
            for dr, dc in MOVE_VECTORS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    danger.add((nr, nc))
        return danger

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

    def _nearest_opponent_distance(
        self,
        cell: Tuple[int, int],
        opponents: Iterable[Tuple[int, int]],
    ) -> int:
        best = float("inf")
        for op in opponents:
            best = min(best, self._manhattan(cell, op))
        return int(best) if best != float("inf") else 9999

    def _control_targets(
        self,
        start: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        danger: Set[Tuple[int, int]],
        grid_size: int,
    ) -> List[Tuple[Tuple[int, int], float]]:
        queue = deque([(start, 0)])
        visited = {start}
        scored: List[Tuple[Tuple[int, int], float]] = []

        while queue:
            cell, dist = queue.popleft()
            if dist > grid_size:
                continue

            if cell != start:
                exits = self._safe_exit_count(cell, blocked, danger, grid_size)
                territory = self._reachable_space(cell, blocked, grid_size, limit=40)
                center = self._distance_from_wall(cell, grid_size)
                hazard_penalty = 2.0 if cell in danger else 0.0
                score = exits * 3.5 + territory * 0.35 + center * 1.2 - hazard_penalty * 2.0 - dist * 0.55
                scored.append((cell, score))

            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size):
                    continue
                if nxt in blocked or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:4]

    def _reachable_space(
        self,
        start: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        grid_size: int,
        limit: int,
    ) -> int:
        queue = deque([start])
        visited = {start}
        count = 0

        while queue and count < limit:
            cell = queue.popleft()
            count += 1
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size):
                    continue
                if nxt in blocked or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return count

    def _random_legal_move(self, position: Tuple[int, int], grid_size: int) -> str:
        legal: List[str] = []
        for move, (dr, dc) in MOVE_VECTORS.items():
            nr, nc = position[0] + dr, position[1] + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                legal.append(move)
        return random.choice(legal or MOVE_NAMES)
