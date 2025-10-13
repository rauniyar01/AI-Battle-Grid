import importlib.util
import os
import sys
import random
from typing import Dict, List
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from battle_arena import BattleArena


# -------------------- Agent Loader --------------------
def load_agent_from_file(filepath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    AgentClass = getattr(module, "Agent", None)
    if AgentClass is None:
        raise ValueError(f"No Agent class found in {filepath}")
    return AgentClass(name)


def find_agent_files(agents_dir: str) -> List[str]:
    py_files = []
    for fname in os.listdir(agents_dir):
        if fname.endswith(".py") and not fname.startswith("_"):
            py_files.append(os.path.join(agents_dir, fname))
    py_files.sort()
    return py_files


# -------------------- Visualization Helpers --------------------
def make_board_image(grid_size, positions, food, name_to_id):
    """
    Produce a 2D array of IDs:
        0   = empty cell
        1..N = agents
        N+1 = food
    """
    board = np.zeros((grid_size, grid_size), dtype=int)
    # place agents
    for name, (r, c) in positions.items():
        board[r, c] = name_to_id[name]
    # place food
    food_value = len(name_to_id) + 1
    for (r, c) in food:
        board[r, c] = food_value
    return board


# -------------------- Animation --------------------
def animate_battle(arena: BattleArena, agent_objs, fps=6, seed=None):
    random.seed(seed)
    name_to_id = {name: i + 1 for i, name in enumerate(arena.agent_names)}  # 1..N
    N = len(name_to_id)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(top=0.88, bottom=0.15)

    # ---- Unified color palette ----
    base_cmap = plt.colormaps["Set1"]  # modern API (no deprecation)
    agent_colors = base_cmap(np.linspace(0, 1, N, endpoint=False))  # N agent colors
    background = np.array([[0.93, 0.93, 0.93, 1.0]])  # light gray
    food_color = np.array([[0.121, 0.466, 0.705, 1.0]])  # blue
    # 0=bg, 1..N=agents, N+1=food
    colors = np.vstack([background, agent_colors, food_color])
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, (N + 1) + 1.5, 1)  # integer bins
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # ---- Initial grid ----
    img = ax.imshow(
        make_board_image(arena.grid_size, arena.positions, arena.food, name_to_id),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        animated=True,
    )
    ax.set_xticks(range(arena.grid_size))
    ax.set_yticks(range(arena.grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)

    # ---- Legend ----
    patches = []
    for i, name in enumerate(name_to_id.keys(), start=1):
        patches.append(mpatches.Patch(color=colors[i], label=name))
    patches.append(mpatches.Patch(color=colors[N + 1], label="Food"))
    ax.legend(
        handles=patches,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.0),
        fontsize=9,
        frameon=False,
    )

    # ---- Scoreboard ----
    score_text = ax.text(
        0.02, -0.08, "", transform=ax.transAxes, ha="left", va="top", fontsize=10
    )

    # ---- Frame update ----
    def update_frame(_):
        if arena.is_over():
            return (img, score_text)

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

        # update grid
        board = make_board_image(arena.grid_size, arena.positions, arena.food, name_to_id)
        img.set_data(board)

        alive_names = sorted(list(arena.alive))
        scoreboard = " | ".join(
            f"{n}:{arena.scores.get(n,0)}" for n in sorted(arena.agent_names)
        )
        ax.set_title(
            f"Turn {arena.turn}/{arena.max_turns}   Alive: {len(alive_names)}   {', '.join(alive_names)}",
            fontsize=11,
        )
        score_text.set_text(f"Scores: {scoreboard}")
        return (img, score_text)

    interval = int(1000 / max(1, fps))
    ani = animation.FuncAnimation(
        fig, update_frame, interval=interval, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()
    return ani


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Run the AI Battle on the Grid.")
    parser.add_argument("--agents-dir", default="agents", help="Directory with agent .py files.")
    parser.add_argument("--grid", type=int, default=10, help="Grid size (NxN).")
    parser.add_argument("--turns", type=int, default=100, help="Max number of turns.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--food", type=int, default=3, help="Initial food count.")
    parser.add_argument("--spawn", type=float, default=0.25, help="Food spawn chance per turn (0..1).")
    parser.add_argument("--fps", type=int, default=6, help="Animation frames per second.")
    args = parser.parse_args()

    agents_dir = args.agents_dir
    if not os.path.isdir(agents_dir):
        print(f"Creating '{agents_dir}' directory with an example agent_template.py")
        os.makedirs(agents_dir, exist_ok=True)
        with open(os.path.join(agents_dir, "agent_template.py"), "w") as f:
            f.write("# Copy this file and implement your Agent class here.\n")

    files = find_agent_files(agents_dir)
    if not files:
        print("No agent .py files found in the agents directory.")
        print("Add one or more agent files implementing class Agent(name) with decide_move(game_state).")
        sys.exit(1)

    # Load agents dynamically
    agent_objs = {}
    agent_names = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        agent = load_agent_from_file(fp, stem)
        agent_objs[stem] = agent
        agent_names.append(stem)

    # Initialize arena
    arena = BattleArena(
        grid_size=args.grid,
        agents=agent_names,
        max_turns=args.turns,
        initial_food=args.food,
        food_spawn_chance=args.spawn,
        seed=args.seed,
    )
    arena.reset(agent_names)

    ani = animate_battle(arena, agent_objs, fps=args.fps, seed=args.seed)

    winners = arena.winner()
    if winners is None:
        print("Game not finished?")
    else:
        print("Winner(s):", winners)
        print("Final scores:", arena.scores)


if __name__ == "__main__":
    main()
