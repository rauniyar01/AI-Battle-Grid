from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional
import random

Coord = Tuple[int, int]

MOVE_VECTORS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

VALID_MOVES = set(MOVE_VECTORS.keys())

@dataclass(frozen=True)
class GameState:
    grid_size: int
    you: str
    positions: Dict[str, Coord]     # all agent positions
    food: Set[Coord]                # food coordinates
    alive: Set[str]                 # alive agent names
    scores: Dict[str, int]          # current scores
    turn: int
    max_turns: int

class BattleArena:
    """
    Turn-based grid arena with simultaneous moves, food collection, and collisions.

    Rules:
    - Agents are points on a grid (no tails).
    - Each turn, every ALIVE agent chooses one of: UP, DOWN, LEFT, RIGHT.
    - Moving into walls eliminates the agent.
    - If multiple agents move into the same cell, all those agents are eliminated (head-on).
    - If two agents cross paths (A->B cell, B->A cell), both are eliminated (swap collision).
    - If an agent lands on food, it gains +1 score and the food disappears.
    - New food may spawn with a given probability each turn on empty cells.
    - Winner: highest score at the end (or last remaining agent if others eliminated).
    """
    def __init__(
        self,
        grid_size: int = 10,
        agents: Optional[List[str]] = None,
        max_turns: int = 100,
        initial_food: int = 3,
        food_spawn_chance: float = 0.25,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.agent_names = list(agents or [])
        self.max_turns = max_turns
        self.initial_food = initial_food
        self.food_spawn_chance = food_spawn_chance
        self.rng = random.Random(seed)

        # dynamic state
        self.turn = 0
        self.positions: Dict[str, Coord] = {}
        self.alive: Set[str] = set()
        self.food: Set[Coord] = set()
        self.scores: Dict[str, int] = {}

    # ----------------- Initialization -----------------
    def reset(self, agents: List[str]):
        self.agent_names = list(agents)
        self.turn = 0
        self.alive = set(self.agent_names)
        self.scores = {a: 0 for a in self.agent_names}

        # place agents in distinct random free cells
        occupied = set()
        self.positions = {}
        for a in self.agent_names:
            pos = self._random_free_cell(occupied)
            self.positions[a] = pos
            occupied.add(pos)

        # place initial food
        self.food = set()
        for _ in range(self.initial_food):
            self._spawn_food()

    def _random_free_cell(self, occupied: Set[Coord]) -> Coord:
        while True:
            r = self.rng.randrange(self.grid_size)
            c = self.rng.randrange(self.grid_size)
            if (r, c) not in occupied and (r, c) not in self.food:
                return (r, c)

    def _spawn_food(self):
        occupied = set(self.positions.values()) | self.food
        # If board is full, do nothing
        if len(occupied) >= self.grid_size * self.grid_size:
            return
        cell = self._random_free_cell(occupied)
        self.food.add(cell)

    # ----------------- Turn Mechanics -----------------
    def step(self, moves: Dict[str, str]) -> Dict[str, str]:
        """
        Apply one simultaneous-move step.
        'moves' must contain a move for each currently ALIVE agent.
        Returns a dict of final outcomes for agents this turn:
            'OK', 'ELIMINATED_WALL', 'ELIMINATED_COLLISION', 'ELIMINATED_SWAP'
        """
        self.turn += 1

        # Normalize invalid/missing moves to random valid move to keep game flowing
        sanitized_moves: Dict[str, str] = {}
        for a in list(self.alive):
            mv = moves.get(a, None)
            if mv not in VALID_MOVES:
                mv = self.rng.choice(list(VALID_MOVES))
            sanitized_moves[a] = mv

        # Compute intended new positions
        intended: Dict[str, Coord] = {}
        for a in list(self.alive):
            r, c = self.positions[a]
            dr, dc = MOVE_VECTORS[sanitized_moves[a]]
            intended[a] = (r + dr, c + dc)

        outcomes: Dict[str, str] = {a: "OK" for a in self.agent_names}

        # 1) Wall collisions
        eliminated_wall: Set[str] = set()
        for a, (r, c) in intended.items():
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                eliminated_wall.add(a)
        for a in eliminated_wall:
            outcomes[a] = "ELIMINATED_WALL"

        # 2) Head-on collisions: same target cell by multiple agents
        cell_to_agents: Dict[Coord, List[str]] = {}
        for a, cell in intended.items():
            if a in eliminated_wall:
                continue
            cell_to_agents.setdefault(cell, []).append(a)

        eliminated_collision: Set[str] = set()
        for cell, agents in cell_to_agents.items():
            if len(agents) > 1:
                for a in agents:
                    if a not in eliminated_wall:
                        eliminated_collision.add(a)
                        outcomes[a] = "ELIMINATED_COLLISION"

        # 3) Swap collisions: A moves into B's cell while B moves into A's cell
        eliminated_swap: Set[str] = set()
        if not eliminated_wall:
            for a in list(self.alive):
                if a in eliminated_wall or a in eliminated_collision:
                    continue
                for b in list(self.alive):
                    if b <= a:
                        continue
                    if b in eliminated_wall or b in eliminated_collision:
                        continue
                    # positions before
                    pa = self.positions[a]
                    pb = self.positions[b]
                    # intended positions
                    ia = intended[a]
                    ib = intended[b]
                    if ia == pb and ib == pa:
                        eliminated_swap.add(a)
                        eliminated_swap.add(b)

        for a in eliminated_swap:
            outcomes[a] = "ELIMINATED_SWAP"

        # 4) Update survivors' positions
        survivors = set(self.alive) - eliminated_wall - eliminated_collision - eliminated_swap
        for a in survivors:
            self.positions[a] = intended[a]

        # 5) Food collection
        # If multiple survivors land on food (can't happen due to earlier collision removal), but just in case:
        collected_cells: Set[Coord] = set()
        for a in survivors:
            pos = self.positions[a]
            if pos in self.food:
                self.scores[a] += 1
                collected_cells.add(pos)
        self.food -= collected_cells

        # 6) Randomly spawn new food
        if self.rng.random() < self.food_spawn_chance:
            self._spawn_food()

        # 7) Mark eliminated
        self.alive -= (eliminated_wall | eliminated_collision | eliminated_swap)

        return outcomes

    # ----------------- State Exposure -----------------
    def get_game_state_for(self, you: str) -> GameState:
        """Return a read-only snapshot of the game for agent 'you'."""
        return GameState(
            grid_size=self.grid_size,
            you=you,
            positions=dict(self.positions),
            food=set(self.food),
            alive=set(self.alive),
            scores=dict(self.scores),
            turn=self.turn,
            max_turns=self.max_turns,
        )

    # ----------------- Utility -----------------
    def is_over(self) -> bool:
        if self.turn >= self.max_turns:
            return True
        if len(self.alive) <= 1:
            return True
        return False

    # def winner(self) -> Optional[List[str]]:
    #     """Return list of winner(s) (could be tie), or None if game not over."""
    #     if not self.is_over():
    #         return None
    #     # If only one alive, that's the winner
    #     if len(self.alive) == 1:
    #         return list(self.alive)
    #     # Else decide by score
    #     max_score = max(self.scores.values()) if self.scores else 0
    #     winners = [a for a, s in self.scores.items() if s == max_score]
    #     return winners

    def winner(self):
        """Winner = alive agent with highest score; if all dead, highest score overall."""
        if len(self.alive) > 0:
            alive_scores = {a: self.scores[a] for a in self.alive}
            max_score = max(alive_scores.values())
            return [a for a, s in alive_scores.items() if s == max_score]
        else:
            max_score = max(self.scores.values())
            return [a for a, s in self.scores.items() if s == max_score]

