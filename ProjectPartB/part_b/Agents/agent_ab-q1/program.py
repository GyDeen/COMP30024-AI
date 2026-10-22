# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import (
    Action,
    CascadeAction,
    Direction,
    EatAction,
    MoveAction,
    PlaceAction,
    PlayerColor,
)

from .action import execute_move, pack_move
from .board import BoardState
from .heuristic import end_phase_eval, eval0, eval_L_place
from .search import get_best_move

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

# FLAG for TT
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


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
        self.evaluate = eval0
        self.end_phase_eval = end_phase_eval
        self.place_eval = eval_L_place
        self._turn_count = 0
        self.board = BoardState()
        self.tt = {} # values are (score, depth, flag, best_move)
        if color == PlayerColor.RED:
            print("Testing: I am playing as RED (first player)")
        elif color == PlayerColor.BLUE:
            print("Testing: I am playing as BLUE")
        else:
            raise ValueError(f"unexpected color: {color}")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """
        root_color = RED if self._color == PlayerColor.RED else BLUE
        time_remaining = referee.get("time_remaining")
        
        self.manage_tt_mem()
        if isinstance(time_remaining, (int, float)):
            remaining = float(time_remaining)
        else:
            remaining = 180.0
        if self.board.turn_count < 8:
            return get_best_move(self.board, root_color, self._color == PlayerColor.RED, remaining, self.place_eval, self.evaluate, self.tt)
        return get_best_move(self.board, root_color, self._color == PlayerColor.RED, remaining, self.place_eval, self.evaluate, self.tt)

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
        
    def manage_tt_mem(self):
        """
        Actively manages the dictionary size to prevent exceeding peak memory limits.
        A 64-byte key and a 4-item tuple value means we can store roughly 800,000 to 
        1,000,000 entries safely within 250 MB.
        """
        MAX_ENTRIES = 800000
        keys_to_delete = [
                key for key, entry in self.tt.items() 
                if entry[1] <= 2 
            ]
        
        for key in keys_to_delete:
                del self.tt[key]
                
        # If pruning shallow depths wasn't enough to reduce the size 
        # we can prune more aggressively by removing entries with depth <= 4
        keys_to_delete = [
            key for key, entry in self.tt.items()
            if entry[1] <= 4
        ]

        for key in keys_to_delete:
            del self.tt[key]
