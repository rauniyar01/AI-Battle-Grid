# AI Battle on the Grid – Food Frenzy (Python Starter)

## Quick Start
1. Ensure Python 3.9+ and `matplotlib` installed:
   ```bash
   pip install matplotlib
   ```
2. Run a demo battle with two example agents:
   ```bash
   python run_battle.py --fps 8 --turns 120 --grid 12 --spawn 0.3
   ```

---

## 🧠 Command-Line Arguments

The main script `run_battle.py` supports several parameters that control how the simulation behaves.  
You can adjust these to change grid size, game speed, or food spawn rate.

| Argument  | Meaning                     | Example value | Description                                                                                                                                                                         |
| --------- | --------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--fps`   | **Frames per second**       | `6`           | Controls animation speed (higher = faster visual updates).                                                                                                                          |
| `--turns` | **Maximum number of turns** | `120`         | Game length — each turn all agents move once. Simulation stops after this many turns (or earlier if all agents die).                                                                |
| `--grid`  | **Grid size (NxN)**         | `12`          | Creates a `12 × 12` board. Each cell can hold one agent or a piece of food. Smaller = tighter space, larger = more exploration.                                                     |
| `--spawn` | **Food spawn probability**  | `0.3`         | At every turn, there’s a 30 % chance that a new piece of food will appear at a random empty cell. Higher values keep the board rich in food; lower values make it more competitive. |

---
   
3. Create your own agent:
   - Copy `agent_template.py` into the `agents/` folder with a new filename, e.g. `smart_agent.py`.
   - Implement the `decide_move(self, game_state)` method.
   - Run again to battle all agents found in `agents/`.

## Rules (Summary)
- Grid NxN (default 10x10).
- Agents choose one move each turn from: UP, DOWN, LEFT, RIGHT.
- Wall or agent collisions eliminate the agent.
- Landing on food gives +1 point; food disappears.
- New food has a per-turn spawn chance (default 25%).
- Game ends when only one agent remains or max turns reached.
- Winner is last alive or highest score (Last alive agent has highest precedence over highest score.)

## Tips
- Use pathfinding (BFS/A*) or any other greedy heuristics to chase food.
- Avoid edges and other agents.
- Try adding memory to avoid traps.
- Use randomness sparingly to break ties.

Happy hacking!
