import random
import numpy as np
from battleship_agents.smart_llm_agent_v2 import SmartLLMAgentV2

class SmartProbabilityAgent(SmartLLMAgentV2):
    """
    A deterministic version of SmartLLMAgentV2 that bypasses the LLM.
    
    It uses the exact same state tracking and probability logic but automatically
    selects the highest-probability target instead of asking an LLM to choose.
    
    Includes "Parallel/Blob" logic to handle closed sequences where 
    standard probability maps might return zero.
    """
    def __init__(self, name="Smart-Prob-Agent"):
        # Initialize parent with dummy model/client since we won't use them
        super().__init__(name, model="deterministic", client=None)
        self.agent_type = "heuristic" # Reclassify as heuristic since no AI model is used

    def get_parallel_break_targets(self, board):
        """
        Identifies targets adjacent to 'closed' hit sequences.
        
        When a ship sequence is capped (e.g. 'm h h m') but not sunk, it usually
        implies adjacent/parallel ships. Probability maps often fail here 
        (returning 0) because no single ship fits the exact footprint.
        
        This method returns neighbors of such clusters to force resolution.
        """
        active_hits = []
        rows, cols = 10, 10
        for r in range(rows):
            for c in range(cols):
                if board[r][c] in ['h', 'a']:
                    active_hits.append((r, c))
        
        if not active_hits:
            return []

        priority_targets = set()
        processed = set()
        
        def is_valid(r, c): return 0 <= r < rows and 0 <= c < cols

        for r, c in active_hits:
            if (r, c) in processed:
                continue
            
            # --- Build Horizontal Sequence ---
            h_line = [(r, c)]
            # Don't add to processed yet, wait until we finish checks
            
            # Scan Left
            nc = c - 1
            while is_valid(r, nc) and board[r][nc] in ['h', 'a']:
                h_line.insert(0, (r, nc))
                nc -= 1
            # Scan Right
            nc = c + 1
            while is_valid(r, nc) and board[r][nc] in ['h', 'a']:
                h_line.append((r, nc))
                nc += 1
                
            # Check H Closure
            r1, c1 = h_line[0]; r2, c2 = h_line[-1]
            h_left_closed = c1 == 0 or board[r1][c1-1] in ['m', 's']
            h_right_closed = c2 == 9 or board[r2][c2+1] in ['m', 's']
            h_closed = h_left_closed and h_right_closed

            # --- Build Vertical Sequence ---
            v_line = [(r, c)]
            # Scan Up
            nr = r - 1
            while is_valid(nr, c) and board[nr][c] in ['h', 'a']:
                v_line.insert(0, (nr, c))
                nr -= 1
            # Scan Down
            nr = r + 1
            while is_valid(nr, c) and board[nr][c] in ['h', 'a']:
                v_line.append((nr, c))
                nr += 1
                
            # Check V Closure
            r1, c1 = v_line[0]; r2, c2 = v_line[-1]
            v_top_closed = r1 == 0 or board[r1-1][c1] in ['m', 's']
            v_bottom_closed = r2 == 9 or board[r2+1][c2] in ['m', 's']
            v_closed = v_top_closed and v_bottom_closed

            # --- Detect Parallel/Confused State ---
            # If a line is longer in one direction but CLOSED, it's a parallel candidate.
            # If lengths are equal (single points or squares), it's a parallel candidate.
            is_parallel_candidate = False
            
            if len(h_line) > len(v_line):
                if h_closed: is_parallel_candidate = True
            elif len(v_line) > len(h_line):
                if v_closed: is_parallel_candidate = True
            else:
                # Equal length implies single points or non-linear blobs
                is_parallel_candidate = True

            if is_parallel_candidate:
                # Add all open neighbors of this cluster to targets
                # We simply add neighbors of ALL involved cells to be safe
                cells_to_expand = set(h_line + v_line)
                for tr, tc in cells_to_expand:
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = tr + dr, tc + dc
                        if is_valid(nr, nc) and board[nr][nc] == '.':
                            priority_targets.add((nr, nc))

            # Mark all involved cells as processed so we don't re-evaluate this cluster
            for cell in h_line: processed.add(cell)
            for cell in v_line: processed.add(cell)

        return list(priority_targets)

    async def select_move_async(self, board, remaining):
        """
        Main entry point. Identical setup to SmartLLMAgentV2, but skips the API call.
        """
        # --- 1. STATE RECONCILIATION (Identical to V2) ---
        ship_sank = len(remaining) < len(self.previous_remaining)
        if ship_sank and self.last_move is not None:
            sunk_size = self._compute_sunk_size_from_remaining(remaining, self.previous_remaining)
            if sunk_size:
                self.sunk_deductor(self.last_move, sunk_size)
        
        self.try_resolve_unsolved(remaining)
        self.apply_deductions_to_agent_board()
        self.previous_remaining = remaining.copy()
        self.validate_agent_board()

        # --- 2. PRIORITY CHECK: PARALLELS/BLOBS ---
        # Before doing probability math, check if we are stuck in a "closed" state.
        parallel_targets = self.get_parallel_break_targets(self.agent_board)
        if parallel_targets:
            # If we have parallel targets, pick the one with the most open neighbors 
            # (heuristic to clear space efficiently)
            best_p_move = None
            max_n = -1
            for r, c in parallel_targets:
                n = self.count_unknown_neighbors(self.agent_board, r, c)
                if n > max_n:
                    max_n = n
                    best_p_move = (r, c)
            
            self.last_prompt = "Deterministic - Parallel Break"
            self.last_reasoning = f"Detected closed/parallel structure. Targeting neighbor {best_p_move} to resolve."
            return best_p_move

        # --- 3. CALCULATE PROBABILITIES ---
        display_board = self.create_display_board()
        features = self.compute_smart_llm_probabilities(display_board, remaining)
        coverage_grid = features.get('coverage', np.zeros((10,10)))

        # --- 4. DETERMINE MOVE (Standard Probabilistic) ---
        best_move = self.get_best_probability_move(coverage_grid, self.agent_board)
        
        self.last_prompt = "Deterministic - Probability"
        self.last_reasoning = f"Selected {best_move} based on highest probability score."
        
        if best_move:
            return best_move
        
        # Fallback
        return self.get_fallback_move(coverage_grid, self.agent_board)

    def get_best_probability_move(self, coverage_grid, board):
        """Finds the coordinate with the absolute highest probability."""
        candidates = []
        for r in range(10):
            for c in range(10):
                if board[r][c] == '.': # Only target unknown cells
                    prob = coverage_grid[r][c]
                    # Secondary sort key: number of open neighbors (heuristic for better search)
                    neighbors = self.count_unknown_neighbors(board, r, c)
                    candidates.append((prob, neighbors, (r, c)))
        
        if not candidates:
            return None
            
        # Sort by Probability (Desc), then Neighbors (Desc)
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Pick the top one
        return candidates[0][2]