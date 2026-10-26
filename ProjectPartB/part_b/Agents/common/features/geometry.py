"""Board geometry used by feature calculations, independent of agent search."""

CENTER_4 = frozenset({27, 28, 35, 36})
CENTER_CELLS = (27, 28, 35, 36)
CORNERS = (0, 7, 56, 63)
INNER_SQUARE = frozenset(r * 8 + c for r in range(2, 6) for c in range(2, 6))
CENTER_WEIGHTS = [
    -2, -2, -2, -2, -2, -2, -2, -2,
    -2, 2, 2, 2, 2, 2, 2, -2,
    -2, 2, 3, 3, 3, 3, 2, -2,
    -2, 2, 3, 4, 4, 3, 2, -2,
    -2, 2, 3, 4, 4, 3, 2, -2,
    -2, 2, 3, 3, 3, 3, 2, -2,
    -2, 2, 2, 2, 2, 2, 2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2,
]
MANHATTAN = [[abs((a >> 3) - (b >> 3)) + abs((a & 7) - (b & 7))
              for b in range(64)] for a in range(64)]
EDGE_DIST = [min(r, 7 - r, c, 7 - c) for r in range(8) for c in range(8)]
ADJACENT = []
for _index in range(64):
    _row, _column = divmod(_index, 8)
    _neighbors = {}
    if _row > 0:
        _neighbors[-8] = _index - 8
    if _row < 7:
        _neighbors[8] = _index + 8
    if _column > 0:
        _neighbors[-1] = _index - 1
    if _column < 7:
        _neighbors[1] = _index + 1
    ADJACENT.append(_neighbors)

# Preserve the private geometry names used by the compact-L detector.
_CENTER_4 = CENTER_4
_DIST_TO_CENTER = [min(MANHATTAN[i][c] for c in CENTER_CELLS) for i in range(64)]
_ADJ_SET = [set() for _ in range(64)]
for _index, _neighbors in enumerate(ADJACENT):
    for _neighbor in _neighbors.values():
        _ADJ_SET[_index].add(_neighbor)
