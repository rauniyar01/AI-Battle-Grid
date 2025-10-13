import random
from math import inf

class Agent:
    def __init__(self, name: str):
        self.name = name

    def decide_move(self, game_state):
        # Greedy toward nearest food; otherwise random safe move
        my_pos = game_state.positions[self.name]
        food = list(game_state.food)
        moves = ['UP','DOWN','LEFT','RIGHT']
        best_mv = None
        best_dist = inf
        for mv in moves:
            dr, dc = {'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}[mv]
            nr, nc = my_pos[0]+dr, my_pos[1]+dc
            # Skip moves that hit walls
            if not (0 <= nr < game_state.grid_size and 0 <= nc < game_state.grid_size):
                continue
            # Manhattan distance to nearest food after this move
            if food:
                d = min(abs(nr-fr)+abs(nc-fc) for (fr,fc) in food)
            else:
                d = random.random()*10
            if d < best_dist:
                best_dist = d
                best_mv = mv
        return best_mv or random.choice(moves)
