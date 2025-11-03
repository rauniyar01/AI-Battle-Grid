import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ollama import Client, ResponseError

# Valid moves for the environment
_MOVES = ("UP", "DOWN", "LEFT", "RIGHT")
_MOVE_RE = re.compile(r"\b(UP|DOWN|LEFT|RIGHT)\b", re.IGNORECASE)

def _clamp_move(token: str) -> str:
    if not token:
        return random.choice(_MOVES)
    m = _MOVE_RE.search(token)
    return m.group(1).upper() if m else random.choice(_MOVES)

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _step(from_rc: Tuple[int, int], move: str) -> Tuple[int, int]:
    r, c = from_rc
    if move == "UP":
        return (r - 1, c)
    if move == "DOWN":
        return (r + 1, c)
    if move == "LEFT":
        return (r, c - 1)
    return (r, c + 1)  # RIGHT

class Agent:
    def __init__(
        self,
        name: str,
        api_token: str,
        *,
        host: Optional[str] = None,
        model: Optional[str] = None,
        decision_budget_ms: Optional[int] = None,
        http_timeout_s: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.name = name
        if not api_token:
            raise ValueError("api_token is required for Ollama access.")
        self.api_token = api_token
        # Configurable via env without touching code
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        # decision budget in milliseconds
        self.decision_budget_ms = (
            decision_budget_ms
            if decision_budget_ms is not None
            else int(os.getenv("AI_DECISION_BUDGET_MS", "1200"))
        )
        # network timeout in seconds for the HTTP call
        self.http_timeout_s = (
            http_timeout_s
            if http_timeout_s is not None
            else float(os.getenv("AI_HTTP_TIMEOUT_S", "1.1"))
        )
        # deterministic by default
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("AI_TEMPERATURE", "0.0"))
        )

        # Ollama client configuration (token required by new API)
        default_host = "https://ollama.sct.sintef.no"
        self.host = host or os.getenv("OLLAMA_ENDPOINT", default_host)
        client_kwargs = {
            "host": self.host,
            "timeout": self.http_timeout_s,
        }
        client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_token}"}
        self._client = None
        try:
            self._client = Client(**client_kwargs)
        except Exception:
            # Keep lazy fallback; runtime errors are handled when deciding moves.
            self._client = None

    # ------------ utility: serialize a compact, model-friendly state ------------
    def _serialize_state(self, game_state) -> Dict[str, Any]:
        you = game_state.you
        you_rc = tuple(game_state.positions.get(you, (0, 0)))
        others = {
            n: list(rc)
            for n, rc in game_state.positions.items()
            if n != you and n in game_state.alive
        }
        state = {
            "grid_size": game_state.grid_size,
            "you": you,
            "you_pos": list(you_rc),
            "alive": sorted(list(game_state.alive)),
            "positions": others,  # opponents only, reduces noise
            "food": [list(rc) for rc in sorted(game_state.food)],
            "scores": {k: int(v) for k, v in game_state.scores.items()},
            "turn": int(game_state.turn),
            "max_turns": int(game_state.max_turns),
            "valid_moves": list(_MOVES),
        }
        return state

    # ------------ utility: simple safe fallback if AI is slow/unavailable ------------
    def _greedy_safe(self, game_state) -> str:
        """Move toward nearest food while avoiding immediate collisions and walls."""
        grid = game_state.grid_size
        you_rc = tuple(game_state.positions[game_state.you])
        blocked = set(tuple(rc) for rc in game_state.positions.values() if rc is not None)
        # prefer closest food
        foods: List[Tuple[int, int]] = list(sorted(game_state.food, key=lambda rc: _manhattan(you_rc, rc)))
        candidates: List[str] = list(_MOVES)

        def legal(move: str) -> bool:
            nr, nc = _step(you_rc, move)
            # stay inside grid and do not step onto an occupied cell
            return 0 <= nr < grid and 0 <= nc < grid and (nr, nc) not in blocked

        # try food-directed move first
        if foods:
            target = foods[0]
            vr = target[0] - you_rc[0]
            vc = target[1] - you_rc[1]
            pri: List[str] = []
            pri.append("DOWN" if vr > 0 else "UP" if vr < 0 else "")
            pri.append("RIGHT" if vc > 0 else "LEFT" if vc < 0 else "")
            pri = [m for m in pri if m]
            for m in pri:
                if legal(m):
                    return m
        # otherwise pick any legal move
        legal_moves = [m for m in candidates if legal(m)]
        if legal_moves:
            return random.choice(legal_moves)
        # if trapped, move randomly (environment will handle crash)
        return random.choice(_MOVES)

    def _ask_model(self, state: Dict[str, Any]) -> str:
        system = (
            "You control a single agent on a 2D grid. "
            "Decide exactly one move among UP, DOWN, LEFT, RIGHT. "
            "Output only the word UP or DOWN or LEFT or RIGHT. No explanations."
        )
        user = (
            "Game state (JSON):\n"
            + json.dumps(state, separators=(",", ":"))
        )

        if not self._client:
            raise RuntimeError("Ollama client unavailable")

        options = {"temperature": self.temperature}
        try:
            out = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=False,
                options=options,
            )
            if isinstance(out, dict):
                message = out.get("message")
                if isinstance(message, dict):
                    return _clamp_move(message.get("content", "") or "")
                if "content" in out:
                    return _clamp_move(out.get("content", "") or "")
            return _clamp_move("")
        except ResponseError as exc:
            if exc.status_code != 404:
                raise
        # /api/generate fallback via Ollama client
        out = self._client.generate(
            model=self.model,
            system=system,
            prompt=user,
            stream=False,
            options=options,
        )
        if isinstance(out, dict):
            text = out.get("response", "") or out.get("output", "") or ""
            return _clamp_move(text)
        return _clamp_move("")

    # ------------------------------- main API -------------------------------
    def decide_move(self, game_state) -> str:
        """
        Decide next move using the remote AI model within a decision budget.
        If the model is unavailable or exceeds the budget, a safe greedy fallback is used.
        """
        start = time.perf_counter()
        state = self._serialize_state(game_state)

        # try model, respecting the decision budget
        try:
            move = self._ask_model(state)
        except Exception:
            # network or parsing problem, revert to heuristic
            move = None

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if move in _MOVES and elapsed_ms <= self.decision_budget_ms:
            return move

        # timeout or invalid token, use deterministic safe fallback
        return self._greedy_safe(game_state)
