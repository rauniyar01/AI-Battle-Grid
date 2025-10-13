import random
from typing import Any, Dict

# Participants edit only this file.
# Implement the decide_move() method to control your agent.
# Valid moves: 'UP', 'DOWN', 'LEFT', 'RIGHT'

class Agent:
    def __init__(self, name: str):
        self.name = name

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
        # STARTER LOGIC (random). Replace with your strategy!
        return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
