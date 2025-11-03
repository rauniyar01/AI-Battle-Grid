import random
from typing import Dict, Set, Tuple
from collections import deque
import warnings

class Agent:
    def __init__(self, name: str):
        self.name = name

    def decide_move(self, game_state) -> str:
        my_pos = game_state.positions[game_state.you]
        my_pos = game_state.positions[game_state.you]
        grid_size = game_state.grid_size
        occupied = set(pos for name, pos in game_state.positions.items() if name in game_state.alive and name != game_state.you)

        moves = {
            'UP':    (my_pos[0] - 1, my_pos[1]),
            'DOWN':  (my_pos[0] + 1, my_pos[1]),
            'LEFT':  (my_pos[0], my_pos[1] - 1),
            'RIGHT': (my_pos[0], my_pos[1] + 1)
        }

        def get_valid_moves(pos):
            return [move for move, (nr, nc) in {
                'UP':    (pos[0] - 1, pos[1]),
                'DOWN':  (pos[0] + 1, pos[1]),
                'LEFT':  (pos[0], pos[1] - 1),
                'RIGHT': (pos[0], pos[1] + 1)
            }.items()
                if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in occupied]

        def bfs(start, goal, avoid=None):
            queue = deque()
            queue.append((start, []))
            visited = set()
            visited.add(start)
            avoid = avoid or set()
            while queue:
                current, path = queue.popleft()
                if current == goal:
                    return path
                directions = {
                    'UP':    (current[0] - 1, current[1]),
                    'DOWN':  (current[0] + 1, current[1]),
                    'LEFT':  (current[0], current[1] - 1),
                    'RIGHT': (current[0], current[1] + 1)
                }
                for move, (nr, nc) in directions.items():
                    next_pos = (nr, nc)
                    if (0 <= nr < grid_size and 0 <= nc < grid_size and
                        next_pos not in visited and next_pos not in occupied and next_pos not in avoid):
                        queue.append((next_pos, path + [move]))
                        visited.add(next_pos)
            return None

        if not game_state.food:
            valid_moves = get_valid_moves(my_pos)
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return 'UP'

        # Improved: contest food aggressively, break ties in favor of food closer to self
        best_food = None
        best_path = None
        best_dist = None
        my_paths = {}
        for food in game_state.food:
            path = bfs(my_pos, food)
            if path is None:
                continue
            my_paths[food] = path

        for food, my_path in my_paths.items():
            my_dist = len(my_path)
            min_other_dist = None
            for name, pos in game_state.positions.items():
                if name == game_state.you or name not in game_state.alive:
                    continue
                other_path = bfs(pos, food)
                if other_path is not None:
                    dist = len(other_path)
                    if min_other_dist is None or dist < min_other_dist:
                        min_other_dist = dist
            # Contest food if you are as fast or faster, or break ties in your favor
            if min_other_dist is None or my_dist <= min_other_dist:
                if best_food is None or my_dist < best_dist:
                    best_food = food
                    best_path = my_path
                    best_dist = my_dist

        # If all food is strictly closer to another agent, go for the closest anyway
        if best_path:
            return best_path[0]
        elif my_paths:
            closest_food = min(my_paths, key=lambda f: len(my_paths[f]))
            return my_paths[closest_food][0]
        else:
            valid_moves = get_valid_moves(my_pos)
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return 'UP'
