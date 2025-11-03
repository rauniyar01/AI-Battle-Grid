import random
from collections import deque

class Agent:
    def __init__(self, name: str):
        self.name = name
        self.last_moves = deque(maxlen=4)  # short-term memory to avoid oscillation

    def decide_move(self, game_state) -> str:
        grid_size = game_state.grid_size
        my_pos = game_state.positions[game_state.you]
        food = set(game_state.food)
        others = {pos for n, pos in game_state.positions.items() if n != game_state.you}
        moves = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

        def in_bounds(p):
            return 0 <= p[0] < grid_size and 0 <= p[1] < grid_size

        def neighbors(p):
            for dr, dc in moves.values():
                yield (p[0] + dr, p[1] + dc)

        def is_safe(p):
            if not in_bounds(p) or p in others:
                return False
            # also avoid cells next to other agents
            for nbr in neighbors(p):
                if nbr in others:
                    return False
            return True

        # BFS to all reachable food, keep paths
        queue = deque([(my_pos, [])])
        visited = {my_pos}
        food_paths = []
        while queue:
            pos, path = queue.popleft()
            if pos in food and path:
                # Prioritize food farther from other agents
                food_score = sum(1 for nbr in neighbors(pos) if nbr not in others)
                food_paths.append((path, food_score))
            for move, (dr, dc) in moves.items():
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt not in visited and is_safe(nxt):
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))

        # Step 1: Move toward closest reachable food with highest safety score
        if food_paths:
            # Sort by path length first, then by food safety score
            best_path = min(food_paths, key=lambda x: (len(x[0]), -x[1]))[0]
            next_move = best_path[0]
        else:
            # Step 2: No food reachable safely, pick move maximizing reachable space
            def reachable_area(start, limit=200):
                q = deque([start])
                vis = {start}
                count = 0
                while q and count < limit:
                    p = q.popleft()
                    count += 1
                    for nbr in neighbors(p):
                        if nbr not in vis and in_bounds(nbr) and nbr not in others:
                            vis.add(nbr)
                            q.append(nbr)
                return count

            safe_moves = []
            move_scores = []
            for move, (dr, dc) in moves.items():
                nxt = (my_pos[0] + dr, my_pos[1] + dc)
                if in_bounds(nxt) and nxt not in others:
                    area = reachable_area(nxt)
                    # Dynamic adjacency penalty based on nearby agents
                    adj = sum(1 for nbr in neighbors(nxt) if nbr in others)
                    score = area - adj * (2 + len(others) / 10)  # Adjust penalty dynamically
                    move_scores.append((move, score))
                    safe_moves.append(move)

            if safe_moves:
                # avoid immediate back-and-forth
                for prev in reversed(self.last_moves):
                    if prev in safe_moves:
                        safe_moves.remove(prev)
                        break
                # pick top-scoring move (random among ties)
                top_score = max(score for mv, score in move_scores if mv in safe_moves)
                top_moves = [mv for mv, s in move_scores if s == top_score]
                next_move = random.choice(top_moves)
            else:
                # cornered: pick any legal move
                legal_moves = [m for m, (dr, dc) in moves.items()
                               if in_bounds((my_pos[0]+dr, my_pos[1]+dc))]
                next_move = random.choice(legal_moves) if legal_moves else random.choice(list(moves.keys()))

        self.last_moves.append(next_move)
        return next_move