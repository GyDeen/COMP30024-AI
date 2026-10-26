# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction

from Agents.common.board import BoardState
from Agents.common.action import pack_move, execute_move
from .search import get_best_move
from .heuristic import eval0, eval_L_place

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
        self.place_eval = eval_L_place
        self._turn_count = 0
        self.board = BoardState()
        self.tt = {} # values are (score, depth, flag, best_move)
        self.recent_board_hashes = []
        self.recent_board_set = set()
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
        
        self.manage_tt_mem()
        if isinstance(time_remaining, (int, float)):
            remaining = float(time_remaining)
        else:
            remaining = 180.0
        if self.board.turn_count < 8:
            return get_best_move(self.board, root_color, self._color == PlayerColor.RED, remaining, self.place_eval, self.evaluate, self.tt, self.recent_board_set)
        return get_best_move(self.board, root_color, self._color == PlayerColor.RED, remaining, self.place_eval, self.evaluate, self.tt, self.recent_board_set)

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

        # Record recent full-board states. The search uses this at the root to
        # avoid choosing a move that recreates a recent position, which is what
        # caused the late-game draw loop.
        h = self.board.get_hashable()
        self.recent_board_hashes.append(h)
        self.recent_board_set.add(h)
        if len(self.recent_board_hashes) > 12:
            old = self.recent_board_hashes.pop(0)
            if old not in self.recent_board_hashes:
                self.recent_board_set.discard(old)
        
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