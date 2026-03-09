# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from sys import stdin
from .core import Coord, CellState, PlayerColor, Action, MoveAction, EatAction, CascadeAction
from .program import search

# WARNING: Please *do not* modify any of the code in this file, as this could
#          break things in the submission environment. Failed test cases due to
#          modification of this file will not receive any marks.
#
#          To implement your solution you should modify the `search` function
#          in `program.py` instead, as discussed in the specification.

SOLUTION_PREFIX = "$SOLUTION"


def parse_input(input: str) -> dict[Coord, CellState]:
    """
    Parse CSV input into a board state dictionary. Format uses R3, B2, etc.
    for stacks (color prefix + height), or empty/space for empty cells.
    """
    state = {}

    try:
        for r, line in enumerate(input.strip().split("\n")):
            if line.startswith("#") or line.strip() == "":
                continue
            for c, cell_str in enumerate(line.split(",")):
                cell_str = cell_str.strip()
                if cell_str == "":
                    continue
                if cell_str[0] in ("R", "r"):
                    color = PlayerColor.RED
                elif cell_str[0] in ("B", "b"):
                    color = PlayerColor.BLUE
                else:
                    continue
                height = int(cell_str[1:])
                state[Coord(r, c)] = CellState(color, height)

        return state

    except Exception as e:
        print(f"Error parsing input: {e}")
        exit(1)


def format_action(action: Action) -> str:
    """
    Format an action for output. Uses Direction.name (e.g. Down, Up, Left, Right).
    """
    match action:
        case MoveAction(coord, direction):
            return f"MOVE({coord}, {direction.name})"
        case EatAction(coord, direction):
            return f"EAT({coord}, {direction.name})"
        case CascadeAction(coord, direction):
            return f"CASCADE({coord}, {direction.name})"
        case _:
            return str(action)


def print_result(sequence: list[Action] | None):
    """
    Print the given action sequence, one action per line, or "NOT_FOUND" if no
    sequence was found.
    """
    if sequence is not None:
        for action in sequence:
            print(f"{SOLUTION_PREFIX} {format_action(action)}")
    else:
        print(f"{SOLUTION_PREFIX} NOT_FOUND")


def main():
    """
    Main entry point for program.
    """
    input = parse_input(stdin.read())
    sequence: list[Action] | None = search(input)
    print_result(sequence)


if __name__ == "__main__":
    main()
