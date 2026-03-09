# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import Coord, CellState, PlayerColor, BOARD_N


def apply_ansi(
    text: str,
    bold: bool = False,
    color: str | None = None
):
    """
    Wraps some text with ANSI control codes to apply terminal-based formatting.
    Note: Not all terminals will be compatible!
    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{text}\033[0m"


def render_board(
    board: dict[Coord, CellState],
    ansi: bool = False
) -> str:
    """
    Visualise the board via a multiline ASCII string, including optional ANSI
    styling for terminals that support this. Cells are displayed as R3, B2, etc.
    """
    output = ""
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board.get(Coord(r, c), CellState())
            if cell.is_stack:
                color_char = "R" if cell.color == PlayerColor.RED else "B"
                text = f"{color_char}{cell.height}"
                if ansi:
                    ansi_color = "r" if cell.color == PlayerColor.RED else "b"
                    output += apply_ansi(f"{text:>3}", color=ansi_color)
                else:
                    output += f"{text:>3}"
            else:
                output += " . "
            output += " "
        output += "\n"
    return output
