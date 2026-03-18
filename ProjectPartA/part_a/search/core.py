# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from dataclasses import dataclass
from enum import Enum
from typing import Generator

# WARNING: Please *do not* modify any of the code in this file, as this could
#          break things in the submission environment. Failed test cases due to
#          modification of this file will not receive any marks.
#
#          To implement your solution you should modify the `search` function
#          in `program.py` instead, as discussed in the specification.

BOARD_N = 8


@dataclass(frozen=True, slots=True)
class Vector2:
    """
    A simple 2D vector "helper" class with basic arithmetic operations
    overloaded for convenience.
    """
    r: int
    c: int

    def __lt__(self, other: 'Vector2') -> bool:
        return (self.r, self.c) < (other.r, other.c)

    def __hash__(self) -> int:
        return hash((self.r, self.c))

    def __str__(self) -> str:
        return f"Vector2({self.r}, {self.c})"

    def __add__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r + other.r, self.c + other.c)

    def __sub__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r - other.r, self.c - other.c)

    def __neg__(self) -> 'Vector2':
        return self.__class__(self.r * -1, self.c * -1)

    def __mul__(self, n: int) -> 'Vector2':
        return self.__class__(self.r * n, self.c * n)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.r
        yield self.c


class Direction(Enum):
    """
    An `enum` capturing the four cardinal directions on the board.
    """
    Down  = Vector2(1, 0)
    Up    = Vector2(-1, 0)
    Left  = Vector2(0, -1)
    Right = Vector2(0, 1)

    def __str__(self) -> str:
        return {
            Direction.Down:  "[↓]",
            Direction.Up:    "[↑]",
            Direction.Left:  "[←]",
            Direction.Right: "[→]",
        }[self]

    def __iter__(self):
        return iter(self.value)

    def __getattribute__(self, __name: str) -> int:
        match __name:
            case "r":
                return self.value.r
            case "c":
                return self.value.c
            case _:
                return super().__getattribute__(__name)


@dataclass(order=True, frozen=True)
class Coord(Vector2):
    """
    A specialisation of the `Vector2` class, representing a coordinate on the
    game board. This class also enforces that the coordinates are within the
    bounds of the game board.
    """

    def __post_init__(self):
        if not (0 <= self.r < BOARD_N) or not (0 <= self.c < BOARD_N):
            raise ValueError(f"Out-of-bounds coordinate: {self}")

    def __str__(self):
        return f"{self.r}-{self.c}"

    def __add__(self, other: 'Direction|Vector2') -> 'Coord':
        return self.__class__(
            (self.r + other.r),
            (self.c + other.c),
        )

    def __sub__(self, other: 'Direction|Vector2') -> 'Coord':
        return self.__class__(
            (self.r - other.r),
            (self.c - other.c)
        )


class PlayerColor(Enum):
    """
    An `enum` capturing the two player colours.
    """
    RED = 0
    BLUE = 1

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class CellState:
    """
    A structure representing the state of a cell on the game board. A cell can
    be empty or contain a stack of tokens of a given player colour and height.
    """
    color: PlayerColor | None = None
    height: int = 0

    def __post_init__(self):
        if self.color is None and self.height != 0:
            raise ValueError("Empty cell cannot have non-zero height")
        if self.color is not None and self.height <= 0:
            raise ValueError("Stack must have positive height")

    @property
    def is_empty(self) -> bool:
        return self.color is None

    @property
    def is_stack(self) -> bool:
        return self.color is not None

    def __str__(self):
        if self.is_empty:
            return "."
        color_char = "R" if self.color == PlayerColor.RED else "B"
        return f"{color_char}{self.height}"


@dataclass(frozen=True, slots=True)
class MoveAction:
    """
    A dataclass representing a "move action", which moves a stack one cell
    in a cardinal direction (up, down, left, or right).
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"MOVE({self.coord}, {self.direction})"


@dataclass(frozen=True, slots=True)
class EatAction:
    """
    A dataclass representing an "eat action", which captures an adjacent
    enemy stack. Requires attacker height >= target height.
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"EAT({self.coord}, {self.direction})"


@dataclass(frozen=True, slots=True)
class CascadeAction:
    """
    A dataclass representing a "cascade action", which spreads a stack's tokens
    in a cardinal direction, potentially pushing other stacks.
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"CASCADE({self.coord}, {self.direction})"


Action = MoveAction | EatAction | CascadeAction
