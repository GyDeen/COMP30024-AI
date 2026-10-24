"""Byte board shared by all agents; the baseline turn counter is retained."""

BOARD_N = 8
RED_FLAG = 16
BLUE_FLAG = 32
HEIGHT_MASK = 15



class BoardState:
    __slots__ = ['state', 'red_tokens', 'blue_tokens', 'turn_count']

    def __init__(self, initial_bytes=None):
        if initial_bytes is not None:
            # Clone an existing board
            self.state = bytearray(initial_bytes.state)
            self.red_tokens = initial_bytes.red_tokens
            self.blue_tokens = initial_bytes.blue_tokens
            self.turn_count = initial_bytes.turn_count
        else:
            # Create an empty 64-cell byte board
            self.state = bytearray(64)
            self.red_tokens = 0
            self.blue_tokens = 0
            self.turn_count = 0

    def get_height(self, idx: int) -> int:
        """Returns the height of the stack at the given 1D index (0-63)."""
        return self.state[idx] & HEIGHT_MASK

    def is_red(self, idx: int) -> bool:
        """Returns True if the cell is controlled by Red."""
        return bool(self.state[idx] & RED_FLAG)

    def is_blue(self, idx: int) -> bool:
        """Returns True if the cell is controlled by Blue."""
        return bool(self.state[idx] & BLUE_FLAG)

    def is_empty(self, idx: int) -> bool:
        return self.state[idx] == 0
    
    def set_cell(self, idx: int, is_red: bool, height: int):
        """Creates a completely new stack at a cell."""
        flag = RED_FLAG if is_red else BLUE_FLAG
        self.state[idx] = flag | height

    def clear_cell(self, idx: int):
        """Eliminate a stack completely."""
        if self.is_red(idx):
            self.red_tokens -= self.get_height(idx)
        elif self.is_blue(idx):
            self.blue_tokens -= self.get_height(idx)
            
        self.state[idx] = 0

    def spawn_stack(self, idx: int, is_red: bool):
        """Creates a new stack at the specified index."""
        if is_red:
            self.red_tokens += 3
        else:
            self.blue_tokens += 3

        self.state[idx] = (RED_FLAG if is_red else BLUE_FLAG) | 3

    def get_hashable(self) -> bytes:
        """Converts the mutable bytearray into immutable bytes for hashing."""
        return bytes(self.state)
    
    def lift_stack(self, idx: int) -> int:
        """Removes the stack from the cell and returns its value. Does NOT deduct tokens."""
        val = self.state[idx]
        self.state[idx] = 0
        return val

    def place_stack(self, idx: int, value: int):
        """Places a raw bitwise value directly into a cell."""
        self.state[idx] = value
        
    def render(self):
        """Visualise current board"""
        output = ""
        for r in range(8):
            for c in range(8):
                idx = r * 8 + c
                if self.is_empty(idx):
                    output += ".  "
                else:
                    color_char = "R" if self.is_red(idx) else "B"
                    height = self.get_height(idx)
                    output += f"{color_char}{height} "
            output += "\n"
        print(output)
        
    
    def increment_round(self):
        """Increments the turn count by 1."""
        self.turn_count += 1
