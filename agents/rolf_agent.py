import random
from collections import deque

# 100 runs:
# RTR_CGTP5: avg=12.71, std=5.55, min=0, max=25
# greedy_agent: avg=13.91, std=6.28, min=0, max=25
# random_agent: avg=1.17, std=1.03, min=0, max=4

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
                food_paths.append(path)
            for move, (dr, dc) in moves.items():
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt not in visited and is_safe(nxt):
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))

        # Step 1: Move toward closest reachable food
        if food_paths:
            # pick the path with minimal length
            best_path = min(food_paths, key=len)
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
                    # adjacency penalty
                    adj = sum(1 for nbr in neighbors(nxt) if nbr in others)
                    score = area - adj * 2  # weight adjacency penalty
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
