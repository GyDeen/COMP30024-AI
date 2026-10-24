# Compact Cascade game extraction

The extractor validates a saved referee log and stores one compact game in a
NumPy `.npz` archive. It uses the baseline agent's board encoding, with constants
from `Agents.common.board`. Extraction does not instantiate a playing agent.

## 1. The complete representation

Loading a game gives one dictionary with four fields:

```python
game = {
    "game_id": "baseline-selfplay-001",
    "winner": "DRAW",                  # RED, BLUE or DRAW
    "ending_reason": "threefold_repetition",
    "states": states,                  # NumPy uint8 array, shape (N, 64)
}
```

`states[0]` is the empty board. `states[t]` is the board after `t` accepted
actions. The last state is terminal. A game with 28 actions has 29 states.
Every snapshot is kept in chronological order, including placement positions.

The representation assumes complete games starting from the empty board with
Red moving first. That is the format produced by this referee and enforced
by the replay. It is not a format for isolated arbitrary starting positions.

## 2. How each board is encoded

Each board is a flat array of 64 unsigned one-byte integers. Cell order is
row-major: `state[row * 8 + column]`. It matches the baseline's `bytearray`.

| Cell | Encoding | Example |
|---|---|---|
| Empty | `0` | `0` |
| Red stack | `16 | height` | Height three becomes `19` |
| Blue stack | `32 | height` | Height three becomes `35` |

Decode a cell using the same operations as the agent:

```python
height = int(cell) & 15
is_red = bool(int(cell) & 16)
is_blue = bool(int(cell) & 32)
```

The colour flags occupy different bits from the height. The low four bits
hold heights up to fifteen. Cascade gives each player at most twelve tokens,
so this is sufficient for legal game states. The exporter also checks that
each nonempty height fits before encoding it.

`np.uint8` stores actual bytes, rather than a list of separate Python integer
objects. Each board therefore needs exactly 64 bytes of array data. The array
can be viewed as an eight-by-eight board with `state.reshape(8, 8)`; this does
not change the encoding.

The `.npz` file compresses the array and stores the three strings once.
The file has some archive/array-header overhead, so its size is not simply the
number of board bytes. It stores plain numerical and string arrays, not a
pickled dictionary. Loading uses `allow_pickle=False`.

## 3. Information recovered from the sequence

The position index replaces several fields that were repeated in JSONL:

| Information | Recovery |
|---|---|
| Completed actions | Position index `t` |
| Side to move | Red for even `t`, Blue for odd `t` |
| Phase | Placement for `t < 8`; play otherwise |
| Completed play turns | `max(0, t - 8)` |
| Token totals | Sum decoded heights for each colour |
| Stack counts | Count cells with each colour flag |
| Repetition history | Earlier states together with their inferred side to move |
| Terminal state | Last state in this complete, validated game |
| Game length | `len(states) - 1` actions |

For repetition, count only play-phase states, starting at index eight. Compare
both the board contents and the side to move. A current repetition count alone
cannot replace full history when searching future moves.

The original log retains the actions and event timestamps. They are not copied
into this dataset. When shuffling positions for learning, keep each selected
position's game ID and index so its context and split membership remain known.

## 4. What each function does and why

### `read_log(path)`

Reads the referee's tab-separated events into small `LogEvent` objects. It
checks event names, actor names, argument counts and timestamps. The original
line number is kept for error messages. This is temporary parsing information;
it is not written into every saved position.

### `parse_action(text)`

Converts a display string such as `MOVE(2-3, [→])` into a referee `MoveAction`
with coordinate `(2, 3)` and direction `Right`. It recognises PLACE, MOVE, EAT
and CASCADE without executing log text. Parsing checks syntax; the referee
checks whether the move is legal on the current board.

### `snapshot_board(board)`

Creates a fresh `np.uint8` array of length 64. It reads cells through the
referee's board interface and encodes each occupied cell with its colour flag
and height. Empty cells remain zero. Copying each snapshot prevents later
board mutations from changing historical positions. This function returns
only the array, with no position dictionary or cached counters.

### `ending_reason(board)`

Derives elimination, repetition, stalemate or turn-limit endings in the
referee's precedence order. It reuses the referee's private
`_is_threefold_repetition()` and `_has_legal_actions()` helpers, because no public
ending-reason property exists. These two calls are the dependencies to review
if the referee is replaced. The extractor does not keep another repetition
history or implement another move generator.

### `extract_game(log_path, game_id)`

Starts with an empty referee `Board`, then replays the log in order:

1. Check the start and player initialisation events.
2. Check that each turn number and player colour match the reconstructed board.
3. Parse each action and apply it with `board.apply_action(action)`.
4. Require the following `board_update`, then save a snapshot.
5. Verify that the final board's winner agrees with the logged winner.
6. Join the snapshots into one array with `np.stack(states)` and return the
   four-field game dictionary.

The `board_update` confirmation matters: the referee logs a requested action
BEFORE checking and applying it. An action request alone is not evidence of an
accepted transition. The replay also checks legality through the actual rules.

Incomplete games, forfeits, illegal actions, missing turns and inconsistent
results raise errors. Extraction completes before an output file is opened.

### `write_game(game, output_path)`

Uses `np.savez_compressed()` to save the array and metadata together. The output
must have a `.npz` extension. Opening with `"xb"` refuses to overwrite an existing
file. The dictionary is unpacked into named arrays; it is not pickled as an
opaque Python object.

### `load_game(path)`

Reads the archive with `np.load(..., allow_pickle=False)`. It returns `states`
as a NumPy array and converts the three scalar string arrays back into normal
Python strings. A `with` block closes the archive after reading its contents.

### `main()`

Reads the command-line arguments, calls extraction and saving, then prints the
number of positions and the result. Importing the module does not launch this
command. An input/output error produces a nonzero exit status and a short
message.

## 5. Extract and load a game

Run from the project's `part_b` folder:

```bash
cd "/Users/deenzhou/Documents/Unimelb/COMP30024 AI/ProjectPartB/part_b"
./.venv/bin/python -m DataExtraction.extract_game \
  logs/baseline-selfplay.log \
  --output DataExtraction/output/baseline-selfplay.npz \
  --game-id baseline-selfplay-001
```

NumPy is already in the project's requirements. No additional package is needed.
Use a unique game ID for each distinct game. If omitted, the ID defaults to the
log filename without its extension. Existing outputs are preserved, so choose
a new output filename if the example archive has already been generated.

To inspect the existing example:

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
from DataExtraction.extract_game import load_game

game = load_game(Path("DataExtraction/output/baseline-selfplay.npz"))
states = game["states"]
print(game["game_id"], game["winner"], game["ending_reason"])
print(states.shape, states.dtype, states.nbytes)
print(states[8].reshape(8, 8))
PY
```

Expected array information for the saved self-play game:

```text
(29, 64) uint8 1856
```

## 6. Restore a position to the baseline's board class

```python
from Agents.common.board import BoardState

index = 8
board = BoardState()
board.state = bytearray(game["states"][index].tobytes())
board.turn_count = index
board.red_tokens = sum(board.get_height(i) for i in range(64) if board.is_red(i))
board.blue_tokens = sum(board.get_height(i) for i in range(64) if board.is_blue(i))
is_red_turn = index % 2 == 0
```

Assigning the bytearray restores the cells. Recomputing the two cached token
totals makes the object consistent with the baseline's methods and evaluator.
The side to move is separate because `BoardState` does not store it.

## 7. Checks and next steps

```bash
./.venv/bin/python -m unittest DataExtraction.tests.test_extract_game -v
```

Checks cover all four action types, captures, cascade pushing, repetition,
compatibility with the actual baseline board, compact array size, archive
round-tripping without object arrays, and rejection of malformed logs.

For a play-phase evaluator, select positions starting at index eight and omit
the final terminal position from ordinary nonterminal examples. Keep whole
games together when splitting training, validation and test data.

The winner and ending reason are game metadata, not input features. The winner
describes the game that was played; it is not a deeper-search value or proof
of an optimal outcome. Deeper-search targets will be generated separately.
This one game validates extraction and is not sufficient training data.
