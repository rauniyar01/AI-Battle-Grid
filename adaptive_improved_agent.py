import heapq
import random
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

MOVE_VECTORS: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


class Agent:
    def __init__(self, name: str):
        self.name = name
        self._recent_positions: Deque[Tuple[int, int]] = deque(maxlen=8)
        self._target_food: Optional[Tuple[int, int]] = None
        self._planned_route: Deque[Tuple[int, int]] = deque()

    def decide_move(self, game_state) -> str:
        grid = game_state.grid_size
        my_pos = game_state.positions[self.name]
        my_score = game_state.scores[self.name]
        other_scores = [
            score for name, score in game_state.scores.items() if name != self.name
        ]
        best_other = max(other_scores) if other_scores else my_score
        score_delta = my_score - best_other
        aggression = self._aggression_multiplier(score_delta, game_state.turn, game_state.max_turns)

        self._remember_position(my_pos)

        opponents = {
            name: pos
            for name, pos in game_state.positions.items()
            if name != self.name and name in game_state.alive
        }
        opponent_positions: Set[Tuple[int, int]] = set(opponents.values())

        danger_map = self._build_danger_map(opponent_positions, grid)
        food_cells = set(game_state.food)
        food_threat = self._food_threat(food_cells, opponents)

        if self._target_food not in food_cells:
            self._target_food = None
            self._planned_route.clear()

        cost_field, step_field = self._cost_field(
            start=my_pos,
            grid_size=grid,
            opponent_positions=opponent_positions,
            danger_map=danger_map,
        )

        if not self._target_food and food_cells:
            self._target_food = self._choose_food_target(
                foods=food_cells,
                cost_field=cost_field,
                step_field=step_field,
                food_threat=food_threat,
                aggression=aggression,
                grid_size=grid,
            )
            self._planned_route.clear()

        if self._target_food and self._plan_requires_refresh(
            current=my_pos,
            target=self._target_food,
            opponent_positions=opponent_positions,
            danger_map=danger_map,
            grid_size=grid,
        ):
            self._refresh_plan(
                current=my_pos,
                target=self._target_food,
                opponent_positions=opponent_positions,
                danger_map=danger_map,
                grid_size=grid,
            )

        if self._planned_route and self._planned_route[0] == my_pos:
            self._planned_route.popleft()

        candidate_moves = []
        prev_pos = self._previous_position()
        nearest_food = self._nearest_food_by_cost(cost_field, food_cells)

        for move, (dr, dc) in MOVE_VECTORS.items():
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if not self._in_bounds(nxt, grid):
                continue

            move_score = 0.0
            move_risk = 0.0

            if nxt in opponent_positions:
                move_risk += 220.0 / max(0.6, aggression)

            danger_level = danger_map.get(nxt, 0.0)
            if danger_level > 0.0:
                danger_penalty = 90.0 * danger_level / max(0.6, aggression)
                if nxt in food_cells:
                    opp_margin = food_threat.get(nxt, 6) - 1
                    if opp_margin >= 1:
                        danger_penalty *= 0.35
                    elif opp_margin == 0:
                        danger_penalty *= 0.6
                move_risk += danger_penalty

            if prev_pos and nxt == prev_pos:
                move_risk += 4.0
            if nxt in self._recent_positions:
                move_risk += 3.0

            exit_score = self._safe_exit_score(
                position=nxt,
                opponent_positions=opponent_positions,
                danger_map=danger_map,
                grid_size=grid,
            )
            move_score += exit_score * 12.0

            move_score += self._distance_from_wall_bonus(nxt, grid) * 1.5

            if nxt in food_cells:
                move_score += 130.0 * aggression
            else:
                if self._target_food:
                    dist_to_target = self._shortest_steps(
                        start=nxt,
                        target=self._target_food,
                        grid_size=grid,
                        cut_off=8,
                    )
                    if dist_to_target is not None:
                        margin = food_threat.get(self._target_food, dist_to_target + 2) - dist_to_target
                        aggressive_bonus = 55.0 * aggression
                        move_score += aggressive_bonus - dist_to_target * 9.0
                        if margin < 0:
                            move_score -= 14.0
                if nearest_food:
                    dist_any = self._shortest_steps(
                        start=nxt,
                        target=nearest_food,
                        grid_size=grid,
                        cut_off=8,
                    )
                    if dist_any is not None:
                        move_score += max(0.0, 25.0 - dist_any * 6.0) * aggression

            if self._planned_route and self._planned_route[0] == nxt:
                move_score += 18.0

            adjacency = self._adjacent_opponent_count(nxt, opponent_positions)
            move_score -= adjacency * 15.0

            total = move_score - move_risk + random.random() * 1e-4
            candidate_moves.append((total, move))

        if not candidate_moves:
            return random.choice(list(MOVE_VECTORS.keys()))

        best_score, best_move = max(candidate_moves, key=lambda item: item[0])

        greedy_move = None
        if nearest_food:
            greedy_move = self._prefer_greedy_step(
                start=my_pos,
                target=nearest_food,
                opponent_positions=opponent_positions,
                danger_map=danger_map,
                grid_size=grid,
            )
            if greedy_move:
                for candidate_score, move in candidate_moves:
                    if move == greedy_move and candidate_score >= best_score - 2.0:
                        best_score, best_move = candidate_score, move
                        break

        chosen_vector = MOVE_VECTORS[best_move]
        next_cell = (my_pos[0] + chosen_vector[0], my_pos[1] + chosen_vector[1])
        if self._planned_route and self._planned_route[0] == next_cell:
            self._planned_route.popleft()
        else:
            self._planned_route.clear()

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

    def _aggression_multiplier(self, score_delta: int, turn: int, max_turns: int) -> float:
        if score_delta <= -3:
            base = 1.4
        elif score_delta <= -1:
            base = 1.2
        elif score_delta >= 3:
            base = 0.65
        elif score_delta >= 1:
            base = 0.8
        else:
            base = 1.0
        late = turn / max(1, max_turns)
        if late > 0.75 and score_delta >= 1:
            base *= 0.75
        elif late > 0.75 and score_delta < 0:
            base *= 1.25
        return max(0.55, min(1.5, base))

    def _build_danger_map(
        self,
        opponent_positions: Set[Tuple[int, int]],
        grid_size: int,
        radius: int = 3,
    ) -> Dict[Tuple[int, int], float]:
        danger: Dict[Tuple[int, int], float] = {}
        if not opponent_positions:
            return danger
        decay = {0: 2.5, 1: 1.2, 2: 0.6, 3: 0.25}
        for pos in opponent_positions:
            queue: Deque[Tuple[Tuple[int, int], int]] = deque([(pos, 0)])
            visited: Set[Tuple[int, int]] = {pos}
            while queue:
                cell, dist = queue.popleft()
                weight = decay.get(dist, 0.0)
                if weight > 0:
                    danger[cell] = max(danger.get(cell, 0.0), weight)
                if dist >= radius:
                    continue
                for dr, dc in MOVE_VECTORS.values():
                    nxt = (cell[0] + dr, cell[1] + dc)
                    if nxt in visited or not self._in_bounds(nxt, grid_size):
                        continue
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return danger

    def _food_threat(
        self,
        foods: Set[Tuple[int, int]],
        opponents: Dict[str, Tuple[int, int]],
    ) -> Dict[Tuple[int, int], int]:
        threat: Dict[Tuple[int, int], int] = {}
        if not foods:
            return threat
        for food in foods:
            if opponents:
                threat[food] = min(self._manhattan(pos, food) for pos in opponents.values())
            else:
                threat[food] = 99
        return threat

    def _cost_field(
        self,
        start: Tuple[int, int],
        grid_size: int,
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
        cost_field: Dict[Tuple[int, int], float] = {start: 0.0}
        step_field: Dict[Tuple[int, int], int] = {start: 0}
        frontier: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]

        while frontier:
            cost, cell = heapq.heappop(frontier)
            if cost > cost_field.get(cell, float("inf")) + 1e-9:
                continue
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size):
                    continue
                step_cost = 1.0
                risk_cost = danger_map.get(nxt, 0.0) * 2.8
                occupied_cost = 5.0 if nxt in opponent_positions else 0.0
                edge_cost = 0.35 if min(nxt[0], nxt[1], grid_size - 1 - nxt[0], grid_size - 1 - nxt[1]) == 0 else 0.0
                new_cost = cost + step_cost + risk_cost + occupied_cost + edge_cost
                if new_cost + 1e-9 < cost_field.get(nxt, float("inf")):
                    cost_field[nxt] = new_cost
                    step_field[nxt] = step_field[cell] + 1
                    heapq.heappush(frontier, (new_cost, nxt))
        return cost_field, step_field

    def _choose_food_target(
        self,
        foods: Set[Tuple[int, int]],
        cost_field: Dict[Tuple[int, int], float],
        step_field: Dict[Tuple[int, int], int],
        food_threat: Dict[Tuple[int, int], int],
        aggression: float,
        grid_size: int,
    ) -> Optional[Tuple[int, int]]:
        best_food: Optional[Tuple[int, int]] = None
        best_value: Optional[float] = None

        for food in foods:
            cost = cost_field.get(food)
            steps = step_field.get(food)
            if cost is None or steps is None:
                continue
            opp_steps = food_threat.get(food, steps + 4)
            margin = opp_steps - steps
            margin_bonus = margin * 22.0
            if margin < 0:
                margin_bonus *= 0.4
            value = aggression * 120.0 - cost * 18.0 + margin_bonus
            center_bonus = self._distance_from_wall_bonus(food, grid_size) * 3.0
            value += center_bonus
            if best_value is None or value > best_value:
                best_value = value
                best_food = food
        return best_food

    def _refresh_plan(
        self,
        current: Tuple[int, int],
        target: Tuple[int, int],
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
        grid_size: int,
    ) -> None:
        path = self._find_path(
            start=current,
            goal=target,
            opponent_positions=opponent_positions,
            danger_map=danger_map,
            grid_size=grid_size,
        )
        self._planned_route = deque(path[1:]) if path else deque()

    def _plan_requires_refresh(
        self,
        current: Tuple[int, int],
        target: Tuple[int, int],
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
        grid_size: int,
    ) -> bool:
        if not self._planned_route:
            return True
        if self._planned_route[-1] != target:
            return True
        simulated_position = current
        for cell in self._planned_route:
            if not self._in_bounds(cell, grid_size):
                return True
            if cell in opponent_positions:
                return True
            if danger_map.get(cell, 0.0) > 1.8:
                return True
            if self._manhattan(simulated_position, cell) != 1:
                return True
            simulated_position = cell
        return False

    def _find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
        grid_size: int,
    ) -> Optional[List[Tuple[int, int]]]:
        frontier: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

        while frontier:
            cost, cell = heapq.heappop(frontier)
            if cell == goal:
                break
            if cost > cost_so_far.get(cell, float("inf")) + 1e-9:
                continue
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size):
                    continue
                step_cost = 1.0
                risk_cost = danger_map.get(nxt, 0.0) * 3.5
                occupied_cost = 12.0 if nxt in opponent_positions else 0.0
                new_cost = cost + step_cost + risk_cost + occupied_cost
                if new_cost + 1e-9 < cost_so_far.get(nxt, float("inf")):
                    cost_so_far[nxt] = new_cost
                    came_from[nxt] = cell
                    priority = new_cost + self._manhattan(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))

        if goal not in came_from:
            return None
        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int]] = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    def _safe_exit_score(
        self,
        position: Tuple[int, int],
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
        grid_size: int,
    ) -> float:
        score = 0.0
        for dr, dc in MOVE_VECTORS.values():
            nxt = (position[0] + dr, position[1] + dc)
            if not self._in_bounds(nxt, grid_size):
                continue
            if nxt in opponent_positions:
                continue
            risk = danger_map.get(nxt, 0.0)
            score += max(0.0, 1.2 - risk)
        return score

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

    def _nearest_food_by_cost(
        self,
        cost_field: Dict[Tuple[int, int], float],
        foods: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        nearest: Optional[Tuple[int, int]] = None
        best_cost: Optional[float] = None
        for food in foods:
            cost = cost_field.get(food)
            if cost is None:
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                nearest = food
        return nearest

    def _prefer_greedy_step(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
        grid_size: int,
    ) -> Optional[str]:
        best_move: Optional[str] = None
        best_dist = self._manhattan(start, target)
        best_risk = float("inf")
        for move, (dr, dc) in MOVE_VECTORS.items():
            nxt = (start[0] + dr, start[1] + dc)
            if not self._in_bounds(nxt, grid_size):
                continue
            if nxt in opponent_positions:
                continue
            dist = self._manhattan(nxt, target)
            risk = danger_map.get(nxt, 0.0)
            if dist < best_dist or (dist == best_dist and risk < best_risk):
                best_dist = dist
                best_risk = risk
                best_move = move
        return best_move

    def _shortest_steps(
        self,
        start: Tuple[int, int],
        target: Optional[Tuple[int, int]],
        grid_size: int,
        cut_off: int,
    ) -> Optional[int]:
        if target is None:
            return None
        if start == target:
            return 0
        queue: Deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        visited: Set[Tuple[int, int]] = {start}
        while queue:
            cell, dist = queue.popleft()
            if dist >= cut_off:
                continue
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size) or nxt in visited:
                    continue
                if nxt == target:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None
