from __future__ import annotations

import importlib
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from referee.game import (
    Action,
    CascadeAction,
    Coord,
    Direction,
    EatAction,
    MoveAction,
    PlaceAction,
    PlayerColor,
)

ARENA_DIR = ROOT / "battlefield_arena"
BATTLEFIELD_DIR = ROOT / "battlefield_page"
HOST = "127.0.0.1"
PORT = 8787

DIRS = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
DIR_TO_INDEX = {
    Direction.Up: 0,
    Direction.Down: 1,
    Direction.Left: 2,
    Direction.Right: 3,
}

python_agents: dict[PlayerColor, object] = {}
player_specs: dict[PlayerColor, str] = {
    PlayerColor.RED: "agent:agent",
    PlayerColor.BLUE: "teacher",
}
agent_lock = Lock()


def action_from_index(index: int) -> Action:
    if index < 0 or index >= 832:
        raise ValueError(f"action index out of range: {index}")

    if index < 64:
        return PlaceAction(Coord(index // 8, index % 8))

    if index < 320:
        offset = index - 64
        square = offset // 4
        return MoveAction(Coord(square // 8, square % 8), DIRS[offset % 4])

    if index < 576:
        offset = index - 320
        square = offset // 4
        return EatAction(Coord(square // 8, square % 8), DIRS[offset % 4])

    offset = index - 576
    square = offset // 4
    return CascadeAction(Coord(square // 8, square % 8), DIRS[offset % 4])


def action_to_index(action: Action) -> int:
    if isinstance(action, PlaceAction):
        return action.coord.r * 8 + action.coord.c

    if isinstance(action, MoveAction):
        return 64 + (action.coord.r * 8 + action.coord.c) * 4 + DIR_TO_INDEX[action.direction]

    if isinstance(action, EatAction):
        return 320 + (action.coord.r * 8 + action.coord.c) * 4 + DIR_TO_INDEX[action.direction]

    if isinstance(action, CascadeAction):
        return 576 + (action.coord.r * 8 + action.coord.c) * 4 + DIR_TO_INDEX[action.direction]

    raise ValueError(f"unknown action: {action}")


def player_color(value: str) -> PlayerColor:
    return PlayerColor.RED if value.lower() == "red" else PlayerColor.BLUE


def color_name(color: PlayerColor) -> str:
    return color.name.lower()


def discover_agent_dirs() -> list[str]:
    out = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir() or not path.name.startswith("agent"):
            continue
        if (path / "program.py").is_file():
            out.append(path.name)
    return out


def load_agent(agent_dir: str, color: PlayerColor) -> object:
    if agent_dir not in discover_agent_dirs():
        raise ValueError(f"unknown agent folder: {agent_dir}")
    module = importlib.import_module(f"{agent_dir}.program")
    agent_class = getattr(module, "Agent")
    return agent_class(color)


def player_payload() -> dict:
    agents = [{"id": f"agent:{name}", "label": name} for name in discover_agent_dirs()]
    return {
        "ok": True,
        "players": [
            {"id": "human", "label": "Human"},
            {"id": "teacher", "label": "Teacher"},
            *agents,
        ],
    }


def normalise_player_spec(value: object) -> str:
    spec = str(value)
    if spec in ("human", "teacher"):
        return spec
    if spec.startswith("agent:"):
        agent_dir = spec.split(":", 1)[1]
        if agent_dir in discover_agent_dirs():
            return spec
    raise ValueError(f"unknown player: {spec}")


class ArenaHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/players":
            self.send_json(player_payload())
            return

        if path in ("", "/"):
            self.serve_file(ARENA_DIR / "index.html")
            return

        if path.startswith("/az/"):
            self.serve_file(BATTLEFIELD_DIR / path.lstrip("/"))
            return

        self.serve_file(ARENA_DIR / path.lstrip("/"))

    def do_POST(self) -> None:
        global python_agents, player_specs

        try:
            payload = self.read_json()

            if self.path == "/api/new":
                if "players" in payload:
                    red_spec = normalise_player_spec(payload["players"].get("red", "agent:agent"))
                    blue_spec = normalise_player_spec(payload["players"].get("blue", "teacher"))
                else:
                    agent_color = player_color(str(payload.get("agentColor", "red")))
                    red_spec = "agent:agent" if agent_color == PlayerColor.RED else "teacher"
                    blue_spec = "agent:agent" if agent_color == PlayerColor.BLUE else "teacher"

                next_specs = {
                    PlayerColor.RED: red_spec,
                    PlayerColor.BLUE: blue_spec,
                }
                next_agents: dict[PlayerColor, object] = {}
                for color, spec in next_specs.items():
                    if spec.startswith("agent:"):
                        next_agents[color] = load_agent(spec.split(":", 1)[1], color)
                with agent_lock:
                    player_specs = next_specs
                    python_agents = next_agents
                self.send_json({
                    "ok": True,
                    "players": {color_name(color): spec for color, spec in player_specs.items()},
                })
                return

            if self.path == "/api/notify":
                color = player_color(str(payload["color"]))
                action = action_from_index(int(payload["action"]))
                with agent_lock:
                    for current_agent in python_agents.values():
                        current_agent.update(color, action)
                self.send_json({"ok": True})
                return

            if self.path == "/api/agent_action":
                color = player_color(str(payload["color"]))
                with agent_lock:
                    current_agent = python_agents.get(color)
                    if current_agent is None:
                        raise RuntimeError(f"{color_name(color)} is not a Python agent")
                    action = current_agent.action()
                    index = action_to_index(action)
                self.send_json({"ok": True, "action": index, "text": str(action)})
                return

            if self.path == "/api/agent_eval":
                with agent_lock:
                    current_agent = next(iter(python_agents.values()), None)
                    if current_agent is None:
                        raise RuntimeError("no Python agent is initialised")
                    if not hasattr(current_agent, "model") or current_agent.model is None:
                        self.send_json({
                            "ok": True,
                            "red": 0,
                            "draw": 0,
                            "blue": 0,
                            "turnColor": "red",
                            "policyMax": 0,
                        })
                        return
                    policy_probs, value_probs = current_agent.model.predict(current_agent.encode_board())
                    red = float(value_probs[0])
                    blue = float(value_probs[1])
                    draw = float(value_probs[2])
                    turn_color = current_agent.board.turn_color
                    policy_max = float(policy_probs.max())
                self.send_json({
                    "ok": True,
                    "red": red,
                    "draw": draw,
                    "blue": blue,
                    "turnColor": turn_color.name.lower(),
                    "policyMax": policy_max,
                })
                return
        
            self.send_error(404)

        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[arena] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ArenaHandler)
    print(f"Arena running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
