import numpy as np
from typing import List, Dict
from battleship_agents.board_utils import compute_impossible_mask, num_ships_that_can_fit_here

def make_record(board, remaining, turn, target_index):
    """
    Build a feature record for the given board state and target cell index.
    Matches featureslist.txt exactly.
    """
    features = compute_impossible_mask(board, remaining)
    coverage = features["coverage"]
    coverage_flat = [coverage[r][c] for r in range(10) for c in range(10)]
    impossible_flat = features["impossibleFlat"]

    r, c = divmod(target_index, 10)

    # Helper: check if a neighbor is a hit
    def is_hit(x, y):
        return 0 <= x < 10 and 0 <= y < 10 and board[x][y] == "h"

    # --- Feature calculations matching featureslist.txt ---
    
    # 1. turn
    turn_val = turn
    
    # 2. target_index
    target_idx = target_index
    
    # 3. coverage_mean
    coverage_mean = sum(coverage_flat) / 100.0
    
    # 4. coverage_at_target
    coverage_at_target = coverage[r][c]
    
    # 5. coverage_gradient: difference between target coverage and max neighbor coverage
    neighbor_cov = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        if 0 <= r+dr < 10 and 0 <= c+dc < 10:
            neighbor_cov.append(coverage[r+dr][c+dc])
    coverage_gradient = coverage[r][c] - (max(neighbor_cov) if neighbor_cov else 0)
    
    # 6. impossible_at_target
    impossible_at_target = int(impossible_flat[target_index])
    
    # 7. remaining_sum
    remaining_sum = sum(remaining)
    
    # 8. remaining_count
    remaining_count = len(remaining)
    
    # 9. target_row
    target_row = r
    
    # 10. target_col
    target_col = c
    
    # 11. is_horizontal_hit
    is_horizontal_hit = int(is_hit(r, c-1) or is_hit(r, c+1))
    
    # 11. is_horizontal_hit
    is_horizontal_hit = int(is_hit(r, c-1) or is_hit(r, c+1))
    
    # 12. is_vertical_hit
    is_vertical_hit = int(is_hit(r-1, c) or is_hit(r+1, c))
    
    # 13. is_diagonal_hit
    is_diagonal_hit = int(any(is_hit(r+dr, c+dc) for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]))
    
    # 14. num_ships_that_can_fit_here
    ships_fit = num_ships_that_can_fit_here(board, remaining, r, c)
    
    # 15. max_ship_size_that_fits
    max_ship_size = max(remaining) if remaining else 0

    # --- Return dict matching featureslist.txt exactly ---
    return {
        "turn": turn_val,
        "target_index": target_idx,
        "coverage_mean": coverage_mean,
        "coverage_at_target": coverage_at_target,
        "coverage_gradient": coverage_gradient,
        "impossible_at_target": impossible_at_target,
        "remaining_sum": remaining_sum,
        "remaining_count": remaining_count,
        "target_row": target_row,
        "target_col": target_col,
        "is_horizontal_hit": is_horizontal_hit,
        "is_vertical_hit": is_vertical_hit,
        "is_diagonal_hit": is_diagonal_hit,
        "num_ships_that_can_fit_here": ships_fit,
        "max_ship_size_that_fits": max_ship_size,
    }
