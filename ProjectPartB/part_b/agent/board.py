def sum_red(board: list[int]) -> int:
    """Calculate the sum of all red pieces on the board."""
    return sum(board[i] for i in range(len(board)) if i > 0)


def sum_blue(board: list[int]) -> int:
    """Calculate the sum of all blue pieces on the board."""
    return sum(board[i] for i in range(len(board)) if i < 0)