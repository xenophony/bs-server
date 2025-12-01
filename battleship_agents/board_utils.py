from typing import List, Tuple, Dict

R, C = 10, 10

def to_grid(board: List) -> List[List[str]]:
    """
    Convert board into a 10x10 grid of characters.
    Board can be a list of 10 strings (each length 10) or a list of 10 lists of chars.
    """
    if not isinstance(board, list) or len(board) != R:
        raise ValueError("board must be a list of 10 rows")

    grid = []
    if isinstance(board[0], str):
        for row in board:
            if len(row) != C:
                raise ValueError("each row string must be length 10")
            grid.append(list(row))
    else:
        for row in board:
            if not isinstance(row, list) or len(row) != C:
                raise ValueError("each row list must be length 10")
            grid.append(row[:])
    return grid


def enumerate_valid_placements(grid: List[List[str]], L: int) -> List[List[Tuple[int, int]]]:
    """
    Enumerate all valid placements of a ship of length L on the grid.
    Cells marked 'm' (miss) or 's' (sunk) block placement.
    Cells marked 'h' (hit) must be included if the placement overlaps them.
    """
    placements = []
    for r in range(R):
        for c in range(C):
            # horizontal
            if c + L <= C:
                ok = True
                cells = []
                for cc in range(c, c + L):
                    t = grid[r][cc]
                    if t in ("m", "s"):
                        ok = False
                        break
                    cells.append((r, cc))
                if ok:
                    placements.append(cells)
            # vertical
            if r + L <= R:
                ok = True
                cells = []
                for rr in range(r, r + L):
                    t = grid[rr][c]
                    if t in ("m", "s"):
                        ok = False
                        break
                    cells.append((rr, c))
                if ok:
                    placements.append(cells)
    return placements


def compute_impossible_mask(board: List, remaining: List[int]) -> Dict[str, List]:
    """
    Compute impossible mask given board and remaining ship lengths.
    Returns dict with:
      - coverage (10x10 ints)
      - impossibleGrid (10x10 bools)
      - impossibleFlat (length 100 bools)
    """
    grid = to_grid(board)

    # coverage counts
    counts = [[0]*C for _ in range(R)]
    for L in remaining:
        if not isinstance(L, int) or L <= 0:
            continue
        placements = enumerate_valid_placements(grid, L)
        for p in placements:
            for r, c in p:
                counts[r][c] += 1

    # mark impossible cells
    impossible = [[False]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            cell = grid[r][c]
            if cell != ".":
                # Already shot at (m, h, s) - always impossible
                impossible[r][c] = True
            elif counts[r][c] == 0:
                # Empty but no valid ship placements
                impossible[r][c] = True

    flat = [impossible[r][c] for r in range(R) for c in range(C)]
    return {"coverage": counts, "impossibleGrid": impossible, "impossibleFlat": flat}

def num_ships_that_can_fit_here(board, remaining, r, c):
    """
    Count how many ships in `remaining` could legally fit covering cell (r,c).
    A ship fits if there is enough contiguous '.' or '_' cells in a row/col
    including (r,c) to place it without overlap.
    """
    count = 0
    for size in remaining:
        # Horizontal check
        for start_c in range(c - size + 1, c + 1):
            if 0 <= start_c and start_c + size <= 10:
                segment = [board[r][cc] for cc in range(start_c, start_c + size)]
                if all(cell in [".", "_"] for cell in segment):
                    count += 1
                    break  # only count once per orientation

        # Vertical check
        for start_r in range(r - size + 1, r + 1):
            if 0 <= start_r and start_r + size <= 10:
                segment = [board[rr][c] for rr in range(start_r, start_r + size)]
                if all(cell in [".", "_"] for cell in segment):
                    count += 1
                    break
    return count
