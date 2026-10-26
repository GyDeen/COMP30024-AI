"""Referee adapter for the teaching team's unmodified local JS/WASM agent."""

import atexit
import json
from pathlib import Path
import selectors
import shutil
import subprocess

from referee.agent.resources import ResourceLimitException
from referee.game import PlayerColor

from .actions import action_from_index, action_to_index


LEVEL_BUDGETS = (0, 50, 150, 400, 800, 1600)


class Agent:
    """Level 3 by default; retain worker history by notifying every accepted move."""

    level = 3
    response_timeout = 60

    def __init__(self, color: PlayerColor, **referee):
        node = shutil.which('node')
        if node is None:
            raise RuntimeError('The local teaching agent requires Node.js on PATH')
        self.color = color
        self._turn_color = PlayerColor.RED
        self._request_number = 0
        self._closed = False
        self._process = subprocess.Popen(
            [node, str(Path(__file__).with_name('bridge.mjs'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1,
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            self._request('init', referee, simBudget=LEVEL_BUDGETS[self.level])
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _request(self, kind, referee, **payload):
        if self._closed:
            raise RuntimeError('The teaching agent has been closed')
        self._request_number += 1
        request = {'id': self._request_number, 'type': kind, **payload}
        try:
            self._process.stdin.write(json.dumps(request) + '\n')
            self._process.stdin.flush()
            if not self._selector.select(timeout=self.response_timeout):
                raise TimeoutError(f'The teaching agent did not reply within {self.response_timeout} seconds')
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError('The teaching agent exited; see its console diagnostics')
            response = json.loads(line)
            if response.get('id') != self._request_number:
                raise RuntimeError('The teaching agent returned a mismatched response')
            if not response.get('ok'):
                raise RuntimeError(response.get('error', 'Unknown teaching agent error'))
            remaining = referee.get('time_remaining')
            if remaining is not None and response['cpu_seconds'] > remaining:
                raise ResourceLimitException('exceeded available time (teaching agent Node.js CPU)')
            return response
        except BaseException:
            self.close()
            raise

    def action(self, **referee):
        if self._turn_color != self.color:
            raise ValueError('The teaching agent was asked to move out of turn')
        result = self._request('choose', referee)
        return action_from_index(result['action'])

    def update(self, color, action, **referee):
        if color != self._turn_color:
            raise ValueError('The teaching agent received an out-of-turn update')
        self._request('notify', referee, action=action_to_index(action))
        self._turn_color = PlayerColor.BLUE if color == PlayerColor.RED else PlayerColor.RED

    def close(self):
        """Release the worker on interpreter exit, failure or explicit cleanup."""
        if self._closed:
            return
        self._closed = True
        self._selector.close()
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process.stdout.close()


class Level0Agent(Agent):
    level = 0


class Level1Agent(Agent):
    level = 1


class Level2Agent(Agent):
    level = 2


class Level3Agent(Agent):
    level = 3


class Level4Agent(Agent):
    level = 4


class Level5Agent(Agent):
    level = 5
