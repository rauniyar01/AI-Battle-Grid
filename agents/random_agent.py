import random

class Agent:
    def __init__(self, name: str):
        self.name = name

    def decide_move(self, game_state):
        # Avoid walls a bit by preferring moves that stay inside
        r, c = game_state.positions[self.name]
        moves = ['UP','DOWN','LEFT','RIGHT']
        good = []
        for mv in moves:
            dr, dc = {'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}[mv]
            nr, nc = r+dr, c+dc
            if 0 <= nr < game_state.grid_size and 0 <= nc < game_state.grid_size:
                good.append(mv)
        return random.choice(good or moves)
