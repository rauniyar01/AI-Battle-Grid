"""
Tournament mode for AI Battle Grid.
Runs multiple rounds with different random seeds and reports average scores.
"""

import importlib.util
import os
import random
import numpy as np
from statistics import mean
from battle_arena import BattleArena
from run_battle import load_agent_from_file, find_agent_files

random.seed(42)
np.random.seed(42)


def run_single_match(agent_objs, agent_names, grid=12, turns=120, food=6, spawn=0.3, seed=None):
    """Run one game and return final scores."""
    arena = BattleArena(
        grid_size=grid,
        agents=agent_names,
        max_turns=turns,
        initial_food=food,
        food_spawn_chance=spawn,
        seed=seed,
    )
    arena.reset(agent_names)
    random.seed(seed)

    while not arena.is_over():
        moves = {}
        for name, agent in agent_objs.items():
            if name in arena.alive:
                gs = arena.get_game_state_for(name)
                try:
                    mv = agent.decide_move(gs)
                except Exception:
                    mv = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
                moves[name] = mv
        arena.step(moves)

    return arena.scores


def run_tournament(
    agents_dir="agents",
    rounds=10,
    grid=12,
    turns=120,
    food=6,
    spawn=0.3,
):
    """Run several seeded matches and print averaged leaderboard."""
    files = find_agent_files(agents_dir)
    if not files:
        print("❌ No agents found in:", agents_dir)
        return

    agent_objs = {}
    agent_names = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        agent_objs[name] = load_agent_from_file(fp, name)
        agent_names.append(name)

    # Prepare score tracking
    score_log = {name: [] for name in agent_names}

    print(f"🏁 Starting tournament: {len(agent_names)} agents × {rounds} rounds\n")
    for round_idx in range(1, rounds + 1):
        seed = round_idx
        scores = run_single_match(agent_objs, agent_names, grid, turns, food, spawn, seed)
        print(f" Round {round_idx:2d} | Seed {seed:4d} |", end=" ")
        for n in agent_names:
            print(f"{n}:{scores[n]:2d}", end="  ")
            score_log[n].append(scores[n])
        print("")

    # Compute averages
    print("\n📊 Average Scores (across all rounds):")
    leaderboard = sorted(
        [(n, mean(v)) for n, v in score_log.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    print("-" * 40)
    for rank, (name, avg_score) in enumerate(leaderboard, start=1):
        print(f"{rank:2d}. {name:15s}  avg={avg_score:.2f}")
    print("-" * 40)

    top = leaderboard[0][0] if leaderboard else None
    if top:
        print(f"🏆 Winner of tournament: {top}")
    else:
        print("No winner detected.")


if __name__ == "__main__":
    # Example: run 10 matches with default settings
    run_tournament(rounds=10)
