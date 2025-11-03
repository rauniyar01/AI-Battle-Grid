from math import inf

class Agent:
    def __init__(self, name: str):
        self.name = name

    def decide_move(self, game_state):
        my_pos = game_state.positions[self.name]
        grid_size = game_state.grid_size
        food = list(game_state.food)
        competitors = [pos for agent, pos in game_state.positions.items() if agent != self.name]

        moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        deltas = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

        best_move = None
        best_score = -inf

        for mv in moves:
            dr, dc = deltas[mv]
            nr, nc = my_pos[0] + dr, my_pos[1] + dc

            # Skip moves that hit walls
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue

            # Approximate food influence: nearest food Manhattan distance
            food_score = 0
            if food:
                fr, fc = min(food, key=lambda f: abs(nr - f[0]) + abs(nc - f[1]))
                food_score = 1 / (1 + abs(nr - fr) + abs(nc - fc))

            # Simple competitor avoidance: check minimum distance
            comp_dist = min((abs(nr - cr) + abs(nc - cc)) for (cr, cc) in competitors) if competitors else inf
            comp_score = -1 / (1 + comp_dist)  # closer = worse

            # Soft wall avoidance
            wall_penalty = 1 / (min(nr, nc, grid_size - 1 - nr, grid_size - 1 - nc) + 1)

            # Combine into a single lightweight heuristic
            score = (2.0 * food_score) + (1.5 * comp_score) - (0.5 * wall_penalty)

            if score > best_score:
                best_score = score
                best_move = mv

        # Default if all moves blocked
        return best_move or 'UP'
