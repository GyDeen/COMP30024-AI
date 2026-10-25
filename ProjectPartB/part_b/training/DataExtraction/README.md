# Cascade game and feature extraction

The extractor validates a saved referee log and stores one compact game in a
NumPy `.npz` archive. It uses the baseline agent's board encoding, with constants
from `Agents.common.board`. Extraction does not instantiate a playing agent.

`extract_game.py` is the first preparation stage. `extract_features.py` enriches
its archive with the existing agent's unweighted feature vectors and position
context. Training and notebooks consume these archives; they do not need to
replay referee logs or duplicate the handcrafted feature calculations.

`GameGeneration.generate` calls both stages automatically for completed games,
including [local teaching-agent games](../../Agents/teacher/README.md).
Its `features/` archives have the same format described below; no arena server
or server-specific dataset loader is needed.

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

## 4. Function purposes and data contracts

### `read_log(path)`

Provides the recorded events needed to reconstruct a game. Accepts a referee
log path and returns events in file order, with their timestamps, players,
event names, arguments and source line numbers. Empty or malformed logs are
rejected. These events are temporary replay input, not the extracted dataset.

### `parse_action(text)`

Identifies the action described by a log string. For example,
`MOVE(2-3, [→])` represents moving the stack at `(2, 3)` to the right. Returns
the corresponding referee action object. A valid action description does not
by itself establish that the action is legal in a particular position.

### `snapshot_board(board)`

Represents one position in the dataset's compact format. Accepts a referee
board and returns an independent `uint8` array of 64 cells, in row-major order.
Each cell contains the stack's colour and height in the agent encoding from
section 2. The result describes that position even after the game continues.

### `ending_reason(board)`

Explains why a completed game ended. Accepts the final referee board with its
history and returns `elimination`, `threefold_repetition`, `stalemate` or
`turn_limit`. The referee's precedence determines the reason if conditions
overlap. This is separate from the winner: a repetition draw can occur even
when one player has more tokens.

### `extract_game(log_path, game_id)`

Produces the dataset for one completed game. Accepts a referee log path and
optional game ID, then returns the four-field dictionary shown in section 1.
Without an explicit ID, the log filename stem identifies the game.

The position sequence includes the empty board, the board after every accepted
action, and the final board. A game with 28 actions therefore has 29 positions.
Only complete, naturally finished games with a consistent result are accepted.
The returned dictionary is in memory; this function does not save a file.

### `write_game(game, output_path)`

Persists one extracted game so its positions and result can be used later.
Accepts the dictionary returned by `extract_game()` and a new `.npz` file path.
The archive contains these four named entries:

| Saved entry | What it records |
|---|---|
| `game_id` | Identifier linking all positions to this game |
| `winner` | Final result: `RED`, `BLUE` or `DRAW` |
| `ending_reason` | Elimination, threefold repetition, stalemate or turn limit |
| `states` | A `uint8` array of shape `(N, 64)` containing the entire ordered board history |

Each row of `states` uses the agent's cell encoding: `0` for empty,
`16 | height` for Red, and `32 | height` for Blue. Row zero is the initial
empty board; the last row is terminal. Metadata is stored once per game.

The extracted data does not include actions, timestamps, token totals or
search-evaluation labels. Actions and timestamps remain in the original log;
token totals can be derived from the boards. Evaluation labels are a later
step. This function returns nothing and refuses to replace an existing file.

### `load_game(path)`

Retrieves a saved game for analysis or feature extraction. Accepts the archive
path and returns the same four-field structure as `extract_game()`: three
Python strings and the board-history array. Cell encoding and chronological
position order are preserved.

### `main()`

Makes extraction available from Terminal. Accepts the log path, required
`--output` archive path and optional `--game-id`. Produces the game archive and
reports its number of positions, winner and ending reason. Invalid input or
an output failure is reported with exit status 2.

## 5. Extract and load a game

Run from the project's `part_b` folder:

```bash
cd "/Users/deenzhou/Documents/Unimelb/COMP30024 AI/ProjectPartB/part_b"
./.venv/bin/python -m training.DataExtraction.extract_game \
  logs/baseline-selfplay.log \
  --output training/DataExtraction/output/baseline-selfplay.npz \
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
from training.DataExtraction.extract_game import load_game

game = load_game(Path("training/DataExtraction/output/baseline-selfplay.npz"))
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

## 7. Extract the handcrafted feature dataset

From `part_b`, run:

```bash
./.venv/bin/python -m training.DataExtraction.extract_features \
  training/DataExtraction/output/baseline-selfplay.npz \
  --output training/DataExtraction/output/baseline-selfplay-features.npz
```

The destination must be new. NumPy is the only data-processing dependency;
PyTorch and Jupyter are not needed to extract features. Process each complete
game into its own archive so its game ID and chronological history stay intact.

The extractor calls `Agents.common.features.movement.feature_vector`, which
uses the same `get_stack_maps` and `calc_*` functions imported by `agent_ab-q1`
and the baseline. Feature extraction imports no playing agent or search module.
The eight movement features have this fixed order:

```python
("material", "positioning", "efficiency", "alignment",
 "threaten_enemy", "safety", "hunting", "control")
```

These values are saved before applying profile weights. All eight are computed,
including features with zero weight in the baseline. Both player perspectives
are evaluated independently: some features are asymmetric and turn-dependent,
so Blue's vector cannot generally be obtained by negating Red's.

Each archive contains:

| Entry | Shape / meaning |
|---|---|
| `game_id`, `winner`, `ending_reason` | Original game metadata, stored once |
| `states` | Original `uint8` board history, shape `(N, 64)` |
| `features` | `float32` vectors, shape `(N, 2, 8)` |
| `feature_names` | The eight feature column names, in order |
| `perspectives` | `RED`, `BLUE`: the order of the second feature axis |
| `position_index` | `(N,)`: number of actions completed at each board |
| `side_to_move` | `(N,)`: `RED` or `BLUE`; terminal rows retain the nominal next player |
| `phase` | `(N,)`: `PLACEMENT` or `PLAY` |
| `play_turns`, `remaining_play_turns` | `(N,)`: completed play turns and remaining referee allowance |
| `repetition_state_count` | `(N,)`: occurrences of this board with this side to move, including the current occurrence; zero during placement |
| `is_terminal` | `(N,)`: true only for the final snapshot |
| `baseline_profile_names` | Names of all six profiles from the source agent |
| `baseline_weights` | `(6, 8)`: profile weights aligned with `feature_names` |
| `feature_source`, `schema_version` | Source module name and feature archive format version |

`features[t, 0]` and `features[t, 1]` are Red's and Blue's vectors for
`states[t]`. Features are computed for every snapshot to preserve alignment,
but they describe the movement evaluator. Placement uses a separate baseline
evaluator and should not be mixed into movement training. Terminal rows also
remain available but should be selected deliberately by the trainer.

Context is separate from the eight handcrafted features. Repetition counting
starts with the board after the eighth placement, uses exact board comparisons,
includes only prior/current positions with the same side to move, and never
looks ahead. The full chronological `states` array is preserved because a
current repetition count is not sufficient history for future search.

For the saved sample, the feature shape is `(29, 2, 8)`. Load it in a notebook
started from `part_b`:

```python
from pathlib import Path
from training.DataExtraction.extract_features import load_features

dataset = load_features(
    Path("training/DataExtraction/output/baseline-selfplay-features.npz")
)
mask = (dataset["phase"] == "PLAY") & ~dataset["is_terminal"]
red_features = dataset["features"][mask, 0]
boards = dataset["states"][mask]
indices = dataset["position_index"][mask]
print(dataset["feature_names"])
print(red_features.shape, boards.shape, dataset["winner"])
```

For a notebook saved in `notebooks/`, set its working directory to `part_b`
before these imports and relative paths. The same archive works for random
initialisation or initialisation from a saved baseline profile. The active
`agent_ab-q1` movement evaluator (`eval0`) uses `phalanx_aggressive`; its separate
`endgame` profile is also saved, without choosing how a future model uses it.
These are the agent's raw heuristic weight scales; outcome-target scaling and
any prediction transform remain training choices. No profile evaluation scores
are saved as targets.

`extract_features(game)` returns the enriched dictionary in memory.
`extract_board_features(board, root_color, is_red_turn)` computes one vector
directly from an agent board (`root_color` is 1 for Red, 0 for Blue).
`write_features(dataset, path)` saves a new archive and refuses overwrites;
`load_features(path)` reloads it with `allow_pickle=False`. All entries are plain
numerical or string arrays, with no Python object arrays.

The extractor checks the input archive's shape, dtype, encoding and basic
metadata. It trusts the original `extract_game()` replay for legality and the
correct terminal result; it cannot reconstruct actions from snapshots alone.

## 8. Checks and training boundaries

```bash
./.venv/bin/python -m unittest GameGeneration.tests.test_generation GameGeneration.tests.test_teacher -b -v
```

Checks run real referee games, verify saved boards against replayed logs,
check the feature archive's alignment and shape, and exercise failure handling.
Teacher checks also run the bundled worker locally and validate action encoding.

For a play-phase evaluator, select positions starting at index eight and omit
the final terminal position from ordinary nonterminal examples. Keep whole
games together when splitting training, validation and test data.

The winner and ending reason are game metadata, not model input features. The winner
describes the game that was played; it is not a deeper-search value or proof
of an optimal outcome. Numeric outcome labels and their perspective belong in
the trainer. Both perspectives and all positions from one game must stay in the
same training, validation or test split. This one drawn game validates extraction
and is not sufficient training data.
