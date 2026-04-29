# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

from .board import BoardState
from .action import pack_move, execute_move
from .search import get_best_move
from .heuristic import eval_updated0, eval_place_co2

DIR_TO_OFFSET = {
    Direction.Up: -8,
    Direction.Down: 8,
    Direction.Left: -1,
    Direction.Right: 1
}

ACTION_TO_TYPE = {
    PlaceAction: 0,
    MoveAction: 1,
    EatAction: 2,
    CascadeAction: 3
}

RED = 1
BLUE = 0


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self.evaluate = eval_updated0
        self.place_eval = eval_place_co2
        self._turn_count = 0
        self.board = BoardState()
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """
        root_color = RED if self._color == PlayerColor.RED else BLUE
        time_remaining = referee.get("time_remaining")

        if isinstance(time_remaining, (int, float)):
            remaining = float(time_remaining)
        else:
            remaining = 180.0
        if self.board.turn_count < 8:
            return get_best_move(self.board, root_color, self._color == PlayerColor.RED, remaining, self.place_eval)
        return get_best_move(self.board, root_color, self._color == PlayerColor.RED, remaining, self.evaluate)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        if color == self._color:
            self._turn_count += 1
        
        start_idx = action.coord.r * 8 + action.coord.c
        direction = getattr(action, 'direction', None)
        offset = 0
        if direction is not None:
            offset = DIR_TO_OFFSET.get(direction, 0)
        target_idx = start_idx + offset
        action_type = ACTION_TO_TYPE[type(action)]

        move = pack_move(action_type, start_idx, target_idx)
        self.board = execute_move(self.board, move, color == PlayerColor.RED)

