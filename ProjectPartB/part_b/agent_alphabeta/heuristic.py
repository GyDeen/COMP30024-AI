from agent_alphabeta.board import BoardState

def naive_eval(board: BoardState, is_red: bool) -> int:
    """
    A naive evaluation function that simply counts the number of tokens for each player.
    """
    red_count = board.red_tokens
    blue_count = board.blue_tokens
    if is_red:
        return red_count - blue_count
    else:
        return blue_count - red_count