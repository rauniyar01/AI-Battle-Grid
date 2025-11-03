import heapq
import random
from collections import Counter, deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

MOVE_VECTORS: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
MOVE_ORDER: Sequence[str] = tuple(MOVE_VECTORS.keys())


class Agent:
    def __init__(self, name: str):
        self.name = name
        self._recent_positions: Deque[Tuple[int, int]] = deque(maxlen=8)
        self._target_food: Optional[Tuple[int, int]] = None
        self._primary_plan: Deque[Tuple[int, int]] = deque()
        self._fallback_plan: Deque[Tuple[int, int]] = deque()
        self._last_turn_seen: int = -1

    def decide_move(self, game_state) -> str:
        grid = game_state.grid_size
        my_pos = game_state.positions[self.name]

        if game_state.turn != self._last_turn_seen:
            self._recent_positions.append(my_pos)
            self._last_turn_seen = game_state.turn

        alive_opponents = {
            name: pos
            for name, pos in game_state.positions.items()
            if name != self.name and name in game_state.alive
        }
        opponent_positions: Set[Tuple[int, int]] = set(alive_opponents.values())
        food_cells = set(game_state.food)

        my_score = game_state.scores[self.name]
        other_scores = [score for name, score in game_state.scores.items() if name != self.name]
        best_other = max(other_scores) if other_scores else my_score
        score_delta = my_score - best_other

        aggression = self._aggression_multiplier(
            score_delta=score_delta,
            turn=game_state.turn,
            max_turns=game_state.max_turns,
        )

        opponent_interest = self._predict_opponent_targets(
            opponents=alive_opponents,
            food_cells=food_cells,
        )

        danger_map = self._build_danger_map(
            opponent_positions=opponent_positions,
            grid_size=grid,
            radius=3,
        )

        cost_field, step_field = self._cost_field(
            start=my_pos,
            grid_size=grid,
            opponent_positions=opponent_positions,
            danger_map=danger_map,
        )

        if self._target_food not in food_cells:
            self._target_food = None
            self._primary_plan.clear()
            self._fallback_plan.clear()

        if not self._target_food and food_cells:
            self._target_food = self._choose_food_target(
                foods=food_cells,
                cost_field=cost_field,
                step_field=step_field,
                opponent_interest=opponent_interest,
                aggression=aggression,
                score_delta=score_delta,
            )
            self._primary_plan.clear()
            self._fallback_plan.clear()

        if self._target_food:
            if self._plan_requires_refresh(
                current=my_pos,
                target=self._target_food,
                plan=self._primary_plan,
                opponent_positions=opponent_positions,
                danger_map=danger_map,
                grid_size=grid,
            ):
                self._primary_plan = self._plan_path(
                    start=my_pos,
                    goal=self._target_food,
                    opponent_positions=opponent_positions,
                    danger_map=danger_map,
                    grid_size=grid,
                    risk_bias=3.2,
                )

            if self._plan_requires_refresh(
                current=my_pos,
                target=self._target_food,
                plan=self._fallback_plan,
                opponent_positions=opponent_positions,
                danger_map=danger_map,
                grid_size=grid,
            ):
                self._fallback_plan = self._plan_path(
                    start=my_pos,
                    goal=self._target_food,
                    opponent_positions=opponent_positions,
                    danger_map=danger_map,
                    grid_size=grid,
                    risk_bias=4.8,
                )

        if self._primary_plan and self._primary_plan[0] == my_pos:
            self._primary_plan.popleft()
        if self._fallback_plan and self._fallback_plan[0] == my_pos:
            self._fallback_plan.popleft()

        nearest_food = self._nearest_food_by_cost(cost_field, food_cells)

        candidate_moves = []
        prev_pos = self._previous_position()
        recent_penalty_cells = Counter(self._recent_positions)

        for move in MOVE_ORDER:
            dr, dc = MOVE_VECTORS[move]
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if not self._in_bounds(nxt, grid):
                continue

            score = 0.0
            risk = 0.0

            if nxt in opponent_positions:
                risk += 240.0 / max(0.6, aggression)

            danger_score = danger_map.get(nxt, 0.0)
            if danger_score > 0:
                scaled = 105.0 * danger_score / max(0.6, aggression)
                if nxt in food_cells:
                    opp_margin = opponent_interest.get(("margin", nxt), 0)
                    if opp_margin > 0:
                        scaled *= 0.35
                    elif opp_margin == 0:
                        scaled *= 0.55
                risk += scaled

            if prev_pos and nxt == prev_pos:
                risk += 5.0
            if recent_penalty_cells[nxt] > 1:
                risk += 4.0

            exit_score = self._safe_exit_score(
                position=nxt,
                opponent_positions=opponent_positions,
                danger_map=danger_map,
                grid_size=grid,
            )
            score += exit_score * 11.5

            score += self._distance_from_wall_bonus(nxt, grid) * 1.4

            if nxt in food_cells:
                score += 110.0 * aggression
            else:
                if self._target_food:
                    steps_to_target = step_field.get(nxt)
                    if steps_to_target is None:
                        steps_to_target = self._shortest_steps(
                            start=nxt,
                            target=self._target_food,
                            grid_size=grid,
                            cut_off=8,
                        )
                    if steps_to_target is not None:
                        margin = opponent_interest.get(("arrival", self._target_food), steps_to_target + 2) - steps_to_target
                        score += 50.0 * aggression - steps_to_target * 8.5
                        if margin < 0:
                            score -= 12.0
                if nearest_food:
                    dist_any = self._shortest_steps(
                        start=nxt,
                        target=nearest_food,
                        grid_size=grid,
                        cut_off=7,
                    )
                    if dist_any is not None:
                        score += max(0.0, 22.0 - dist_any * 5.5) * aggression

            if self._primary_plan and self._primary_plan and self._primary_plan[0] == nxt:
                score += 18.0
            elif self._fallback_plan and self._fallback_plan[0] == nxt:
                score += 11.0

            adjacency = self._adjacent_opponent_count(nxt, opponent_positions)
            if adjacency:
                score -= adjacency * (10.0 / max(0.6, aggression))

            total = score - risk + random.random() * 1e-4
            candidate_moves.append((total, move, nxt))

        if not candidate_moves:
            return random.choice(MOVE_ORDER)

        candidate_moves.sort(reverse=True, key=lambda item: item[0])
        best_score, best_move, best_cell = candidate_moves[0]

        greedy_choice = None
        if nearest_food:
            greedy_choice = self._prefer_greedy_step(
                start=my_pos,
                target=nearest_food,
                opponent_positions=opponent_positions,
                danger_map=danger_map,
                grid_size=grid,
            )

        if greedy_choice:
            for score, move, cell in candidate_moves:
                if move == greedy_choice and score >= best_score - 1.5:
                    best_move, best_cell, best_score = move, cell, score
                    break

        if self._primary_plan and self._primary_plan[0] == best_cell:
            self._primary_plan.popleft()
        if self._fallback_plan and self._fallback_plan[0] == best_cell:
            self._fallback_plan.popleft()
        else:
            if best_cell not in list(self._primary_plan)[:3]:
                self._fallback_plan.clear()

        return best_move

    # ---------- Helper methods ----------
    @staticmethod
    def _in_bounds(cell: Tuple[int, int], grid_size: int) -> bool:
        r, c = cell
        return 0 <= r < grid_size and 0 <= c < grid_size

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _previous_position(self) -> Optional[Tuple[int, int]]:
        if len(self._recent_positions) >= 2:
            return self._recent_positions[-2]
        return None

    def _aggression_multiplier(self, score_delta: int, turn: int, max_turns: int) -> float:
        if score_delta <= -3:
            base = 1.45
        elif score_delta <= -1:
            base = 1.25
        elif score_delta >= 3:
            base = 0.6
        elif score_delta >= 1:
            base = 0.8
        else:
            base = 1.0

        phase = turn / max(1, max_turns)
        if phase > 0.75 and score_delta >= 1:
            base *= 0.75
        elif phase > 0.75 and score_delta < 0:
            base *= 1.3

        return max(0.55, min(1.55, base))

    def _predict_opponent_targets(
        self,
        opponents: Dict[str, Tuple[int, int]],
        food_cells: Set[Tuple[int, int]],
    ) -> Dict[Tuple[str, Tuple[int, int]], int]:
        predictions: Dict[Tuple[str, Tuple[int, int]], int] = {}
        if not food_cells:
            return predictions
        for name, pos in opponents.items():
            best_food = None
            best_dist = None
            for food in food_cells:
                dist = self._manhattan(pos, food)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_food = food
            if best_food is not None and best_dist is not None:
                predictions[(name, best_food)] = best_dist
                predictions[("arrival", best_food)] = min(
                    predictions.get(("arrival", best_food), best_dist),
                    best_dist,
                )
                predictions[("margin", best_food)] = predictions.get(("arrival", best_food), best_dist) - best_dist
        return predictions

    def _build_danger_map(
        self,
        opponent_positions: Set[Tuple[int, int]],
        grid_size: int,
        radius: int,
    ) -> Dict[Tuple[int, int], float]:
        danger: Dict[Tuple[int, int], float] = {}
        if not opponent_positions:
            return danger
        decay = {0: 3.0, 1: 1.4, 2: 0.7, 3: 0.3}
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
                risk_cost = danger_map.get(nxt, 0.0) * 2.4
                occupied_cost = 4.5 if nxt in opponent_positions else 0.0
                edge_cost = 0.25 if min(nxt[0], nxt[1], grid_size - 1 - nxt[0], grid_size - 1 - nxt[1]) == 0 else 0.0
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
        opponent_interest: Dict[Tuple[str, Tuple[int, int]], int],
        aggression: float,
        score_delta: int,
    ) -> Optional[Tuple[int, int]]:
        best_food: Optional[Tuple[int, int]] = None
        best_value: Optional[float] = None

        for food in foods:
            cost = cost_field.get(food)
            steps = step_field.get(food)
            if cost is None or steps is None:
                continue
            opp_arrival = opponent_interest.get(("arrival", food), steps + 4)
            margin = opp_arrival - steps
            if score_delta >= 2 and margin < -1:
                continue
            value = aggression * 95.0 - cost * 18.0 + margin * 26.0
            center_bonus = self._distance_from_wall_bonus(food, max(step_field.values(), default=0) + 1) * 2.7
            value += center_bonus
            if best_value is None or value > best_value:
                best_value = value
                best_food = food
        return best_food

    def _plan_requires_refresh(
        self,
        current: Tuple[int, int],
        target: Optional[Tuple[int, int]],
        plan: Deque[Tuple[int, int]],
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
        grid_size: int,
    ) -> bool:
        if target is None:
            return False
        if not plan:
            return True
        if plan[-1] != target:
            return True
        simulated = current
        for cell in plan:
            if not self._in_bounds(cell, grid_size):
                return True
            if cell in opponent_positions:
                return True
            if danger_map.get(cell, 0.0) > 1.6:
                return True
            if self._manhattan(simulated, cell) != 1:
                return True
            simulated = cell
        return False

    def _plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        opponent_positions: Set[Tuple[int, int]],
        danger_map: Dict[Tuple[int, int], float],
        grid_size: int,
        risk_bias: float,
    ) -> Deque[Tuple[int, int]]:
        frontier: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

        while frontier:
            priority, cell = heapq.heappop(frontier)
            if cell == goal:
                break
            if priority > cost_so_far.get(cell, float("inf")) + 1e-9:
                continue
            for dr, dc in MOVE_VECTORS.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self._in_bounds(nxt, grid_size):
                    continue
                step_cost = 1.0
                risk_cost = danger_map.get(nxt, 0.0) * risk_bias
                occupied_cost = 10.0 if nxt in opponent_positions else 0.0
                new_cost = cost_so_far[cell] + step_cost + risk_cost + occupied_cost
                if new_cost + 1e-9 < cost_so_far.get(nxt, float("inf")):
                    cost_so_far[nxt] = new_cost
                    came_from[nxt] = cell
                    heuristic = self._manhattan(nxt, goal)
                    heapq.heappush(frontier, (new_cost + heuristic, nxt))

        if goal not in came_from:
            return deque()
        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int]] = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return deque(path[1:])

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
            score += max(0.0, 1.3 - risk)
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
        return sum(1 for op in opponents if self._manhattan(position, op) == 1)

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
