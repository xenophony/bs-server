import json
import os
import re
import numpy as np
import json_repair
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from battleship_agents.board_utils import compute_impossible_mask, to_grid, enumerate_valid_placements
from collections import Counter
from typing import List, Tuple, Set, Dict
from datetime import datetime


load_dotenv()

class MoveDecision(BaseModel):
    move: str = Field(description="The target cell coordinate (e.g., 'B7').")
    reasoning: str = Field(description="Strategic explanation. State if you are HUNTING a target or SEARCHING.")

class SmartLLMAgentV2:
    def __init__(self, name, model, provider="openai", client=None):
        self.name = name
        self.model = model
        self.provider = provider
        self.agent_type = "llm"
        
        # State
        self.move_history_log = []
        self.last_reasoning = None
        self.last_prompt = None      
        self.last_raw_json = None    
        self.used_fallback = False
        
        # Ship Logic
        self.initial_ships = [5, 4, 3, 3, 2]
        self.previous_remaining = self.initial_ships.copy()
        self.last_move = None
        
        # Agent's internal board (source of truth)
        self.agent_board = [['.' for _ in range(10)] for _ in range(10)]
        
        # Tracking sets
        self.sunk_set = set()          # Confirmed sunk ship coordinates
        self.ambig_set = set()         # Ambiguous hit coordinates
        # NOW: store (coord, size) for ambiguous sinks
        self.unsolved = set()          # {((r,c), size), ...}
        
        # Economics
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # API Failure Tracking
        self.api_failures = []  # List of failure dictionaries
        self.api_call_count = 0  # Total API calls attempted
        self.api_success_count = 0  # Successful API calls

        if client:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ========================================
    # API METADATA CAPTURE
    # ========================================

    def _track_usage(self, usage_obj):
        if usage_obj:
            self.token_usage["prompt_tokens"] += usage_obj.prompt_tokens
            self.token_usage["completion_tokens"] += usage_obj.completion_tokens
            self.token_usage["total_tokens"] += usage_obj.total_tokens



    def log_api_failure(self, error_type, error_message, context=None):
        """Record an API failure with metadata."""
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'turn': len(self.move_history_log) + 1,
            'error_type': error_type.__name__ if isinstance(error_type, type) else str(type(error_type).__name__),
            'error_message': str(error_message),
            'model': self.model,
            'provider': self.provider,
            'last_prompt_preview': self.last_prompt if self.last_prompt else None,  # First 200 chars
            'last_raw_response': self.last_raw_json,
            'context': context or {}
        }
        self.api_failures.append(failure_record)

    def get_failure_metadata(self):
        """Return a summary of API failures for metadata logging."""
        return {
            'total_api_calls': self.api_call_count,
            'successful_calls': self.api_success_count,
            'failed_calls': len(self.api_failures),
            'failure_rate': len(self.api_failures) / self.api_call_count if self.api_call_count > 0 else 0,
            'failures': self.api_failures,
            'fallback_usage': self.used_fallback
        }



    # ========================================
    # AGENT BOARD MANAGEMENT
    # ========================================
    
    def update_agent_board(self, move, result):
        """Update agent's internal board based on game result."""
        r, c = move
        if result.get('hit', False):
            self.agent_board[r][c] = 'h'
        else:
            self.agent_board[r][c] = 'm'
    
    def apply_deductions_to_agent_board(self):
        """Apply sunk ('s') and ambiguous ('a') markers to agent's board."""
        for r, c in self.ambig_set:
            if self.agent_board[r][c] == 'h' and (r, c) not in self.sunk_set:
                self.agent_board[r][c] = 'a'
        for r, c in self.sunk_set:
            if self.agent_board[r][c] in ['h', 'a']:
                self.agent_board[r][c] = 's'

    
    def create_display_board(self):
        """Create display board from agent's internal board."""
        return [row[:] for row in self.agent_board]

    def validate_agent_board(self):
        allowed = {'.', 'm', 'h', 'a', 's'}
        for r, row in enumerate(self.agent_board):
            for c, cell in enumerate(row):
                if cell not in allowed:
                    raise ValueError(f"Invalid cell '{cell}' at {(r,c)}")
        return True

    # ========================================
    # SUNK SHIP DEDUCTION LOGIC
    # ========================================
    
    def _compute_sunk_size_from_remaining(self, remaining, previous):
        counts1 = Counter(previous)
        counts2 = Counter(remaining)
        diff = []
        for item, count in counts1.items():
            for _ in range(count - counts2[item]):
                diff.append(item)
        return diff[0] if diff else None

    def determine_sunk_ship(self, sunk_ship, last_move):
        """
        Determine which coordinates belong to a newly sunk ship of size sunk_ship.
        Uses agent_board and only considers 'h'/'a' (not 's').
        Returns (sunk_coords, ambiguous_coords)
        """
        board = self.agent_board

        def get_vertical_sequence():
            seq = [last_move]
            max_up = [last_move]
            max_down = [last_move]
            up = 1
            down = 1
            while True:
                if last_move[0] + up <= 9 and board[last_move[0] + up][last_move[1]] in ['h', 'a']:
                    seq.append((last_move[0] + up, last_move[1]))
                    if up < sunk_ship:
                        max_up.append((last_move[0] + up, last_move[1]))
                    up += 1
                else:
                    break
            while True:
                if last_move[0] - down >= 0 and board[last_move[0] - down][last_move[1]] in ['h', 'a']:
                    seq.append((last_move[0] - down, last_move[1]))
                    if down < sunk_ship:
                        max_down.append((last_move[0] - down, last_move[1]))
                    down += 1
                else:
                    break
            return seq, max_up, max_down
        
        def get_horizontal_sequence():
            seq = [last_move]
            max_left = [last_move]
            max_right = [last_move]
            right = 1
            left = 1
            while True:
                if last_move[1] + right <= 9 and board[last_move[0]][last_move[1] + right] in ['h', 'a']:
                    seq.append((last_move[0], last_move[1] + right))
                    if right < sunk_ship:
                        max_right.append((last_move[0], last_move[1] + right))
                    right += 1
                else:
                    break
            while True:
                if last_move[1] - left >= 0 and board[last_move[0]][last_move[1] - left] in ['h', 'a']:
                    seq.append((last_move[0], last_move[1] - left))
                    if left < sunk_ship:
                        max_left.append((last_move[0], last_move[1] - left))
                    left += 1
                else:
                    break
            return seq, max_right, max_left
        
        def _is_vert_end():
            y = last_move[0]; x = last_move[1]
            return y+1 > 9 or (y+1 <=9 and  board[y+1][x] not in ['h', 'a']) or y-1 < 0 or (y-1 >= 0 and board[y-1][x] not in ['h', 'a'])

        def _is_horz_end():
            y = last_move[0]; x = last_move[1]
            return x+1 > 9 or (x+1 <=9 and board[y][x+1] not in ['h', 'a']) or x-1 < 0 or (x-1 >= 0 and board[y][x-1] not in ['h', 'a'])
        verticals, max_up, max_down = get_vertical_sequence()
        horizontals, max_right, max_left = get_horizontal_sequence()

        # Case 1: exact match in one direction
        if len(verticals) == sunk_ship and len(horizontals) < sunk_ship:
            return verticals, []
        if len(horizontals) == sunk_ship and len(verticals) < sunk_ship:
            return horizontals, []

        is_vert_end = _is_vert_end()
        is_horz_end = _is_horz_end()

#         print(f"""
# verticals: {verticals} (len={len(verticals)}), horizontals: {horizontals} (len={len(horizontals)})
# is_vert_end: {is_vert_end}, is_horz_end: {is_horz_end}
# max_up: {max_up}, max_down: {max_down}, max_left: {max_left}, max_right: {max_right}
# sunk_ship size: {sunk_ship}
#               """)

        # Case 2: longer at an end
        if len(verticals) > sunk_ship and is_vert_end and len(horizontals) < sunk_ship:
            return (max_up if len(max_up) > 1 else max_down), []
        if len(horizontals) > sunk_ship and is_horz_end and len(verticals) < sunk_ship:
            return (max_right if len(max_right) > 1 else max_left), []
        
        # Case 3: middle of a longer run → ambiguous
        if len(verticals) > sunk_ship and not is_vert_end and len(horizontals) < sunk_ship:
            ambig_seq = set(max_up + max_down)
            return [], list(ambig_seq)
        if len(horizontals) > sunk_ship and not is_horz_end and len(verticals) < sunk_ship:
            ambig_seq = set(max_right + max_left)
            return [], list(ambig_seq)
        
        # Case 4: both directions long enough → ambiguous union
        if len(verticals) >= sunk_ship and len(horizontals) >= sunk_ship:
            ambig_seq = set(max_up + max_down + max_right + max_left)
            return [], list(ambig_seq)
        
        # Fallback
        return [], []

    def update_board_sunk_ambig(self, sunk, ambig):
        self.sunk_set.update(sunk)
        self.ambig_set.update(ambig)
        self.ambig_set.difference_update(self.sunk_set)

    def sunk_deductor(self, last_move, sunk_ship):
        """Process a sinking event using agent_board."""
        # sunk_ship = self._compute_sunk_size_from_remaining(remaining, previous)
        # if sunk_ship is None:
        #     return

        sunk, ambig = self.determine_sunk_ship(sunk_ship, last_move)
        self.update_board_sunk_ambig(sunk, ambig)
        self.apply_deductions_to_agent_board()
        if not sunk and ambig:
            # Could not determine - store (coord, size)
            self.unsolved.add((last_move, sunk_ship))
            
            # print(f"Ambiguous sink of size {sunk_ship} at {last_move}, added to unsolved")
        else:
            # This event resolved
            self.unsolved.discard((last_move, sunk_ship))
            # self.apply_deductions_to_agent_board
            self.try_resolve_unsolved(self.previous_remaining)
            if sunk:
                col_map = "ABCDEFGHIJ"
                coords_str = ", ".join([f"{col_map[c]}{r}" for r, c in sunk])
                # print(f"Identified sunk ship (size {sunk_ship}): {coords_str}")

    def try_resolve_unsolved(self, remaining):
        """
        Re-evaluate previous ambiguous sinks using stored (coord, size),
        with updated board state.
        """
        if not self.unsolved:
            return
        
        col_map = "ABCDEFGHIJ"
        new_unsolved = set()
        
        for (coord, size) in list(self.unsolved):
            # skip if already fully sunk
            if coord in self.sunk_set:
                continue

            sunk, ambig = self.determine_sunk_ship(size, coord)
            if sunk:
                self.sunk_set.update(sunk)
                self.ambig_set.difference_update(sunk)
                coords_str = ", ".join([f"{col_map[c]}{r}" for r, c in sunk])
                # print(f"✅ Resolved previous ambiguous ship of size {size} at {col_map[coord[1]]}{coord[0]} → {coords_str}")
            elif ambig:
                new_unsolved.add((coord, size))
            else:
                print(f"⚠️ No consistent pattern for previous ambiguous ship of size {size} at {col_map[coord[1]]}{coord[0]}, dropping")
        
        self.unsolved = new_unsolved

    def set_cell(self, space, value):
        r, c = space
        self.agent_board[r][c] = value

    # ========================================
    # DISPLAY & FORMATTING
    # ========================================

    def format_board(self, board):
        header = "   A B C D E F G H I J"
        rows = []
        for r in range(10):
            row_str = f"{r}  " + " ".join(board[r])
            rows.append(row_str)
        return header + "\n" + "\n".join(rows)

    def format_prob_map(self, coverage_grid):
        max_val = np.max(coverage_grid)
        if max_val == 0: max_val = 1
        norm_grid = (np.array(coverage_grid) / max_val * 99).astype(int)
        header = "   A  B  C  D  E  F  G  H  I  J"
        rows = []
        for r in range(10):
            row_cells = [f"{x:02d}" for x in norm_grid[r]]
            row_str = f"{r} " + " ".join(row_cells)
            rows.append(row_str)
        return header + "\n" + "\n".join(rows)
    
    def get_top_probability_targets(self, coverage_grid, board, n=6):
        candidates = []
        col_map = "ABCDEFGHIJ"
        for r in range(10):
            for c in range(10):
                if board[r][c] == '.' and coverage_grid[r][c] > 0:
                    neighbor_count = self.count_unknown_neighbors(board, r, c)
                    candidates.append((coverage_grid[r][c], neighbor_count, f"{col_map[c]}{r}"))
        candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
        top_n = candidates[:n]
        if not top_n:
            return "No valid targets."
        return ", ".join([coord for prob, neighbors, coord in top_n])

    def count_unknown_neighbors(self, board, r, c):
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == '.':
                count += 1
        return count

    # ========================================
    # TACTICAL ANALYSIS
    # ========================================

    def get_tactical_hints(self, board):
        active_hits = []
        rows, cols = 10, 10
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'h':
                    active_hits.append((r, c))
        if not active_hits:
            return "No active hits. Proceed to Probability Search."
        
        col_map = "ABCDEFGHIJ"
        def is_valid(r, c): return 0 <= r < rows and 0 <= c < cols
        
        hints = []
        processed = set()
        
        for r, c in active_hits:
            if (r, c) in processed:
                continue
            
            # Build horizontal sequence
            h_line = [(r, c)]
            processed.add((r, c))
            nc = c - 1
            while is_valid(r, nc) and board[r][nc] == 'h':
                h_line.insert(0, (r, nc)); processed.add((r, nc)); nc -= 1
            nc = c + 1
            while is_valid(r, nc) and board[r][nc] == 'h':
                h_line.append((r, nc)); processed.add((r, nc)); nc += 1
            
            # Check if horizontal sequence is "closed" (misses or edges on both ends)
            r1, c1 = h_line[0]; r2, c2 = h_line[-1]
            h_left_closed = c1 == 0 or board[r1][c1-1] in ['m', 's']
            h_right_closed = c2 == 9 or board[r2][c2+1] in ['m', 's']
            h_closed = h_left_closed and h_right_closed
            
            # Build vertical sequence
            v_line = [(r, c)]
            nr = r - 1
            while is_valid(nr, c) and board[nr][c] == 'h':
                v_line.insert(0, (nr, c)); processed.add((nr, c)); nr -= 1
            nr = r + 1
            while is_valid(nr, c) and board[nr][c] == 'h':
                v_line.append((nr, c)); processed.add((nr, c)); nr += 1
            
            # Check if vertical sequence is "closed"
            r1, c1 = v_line[0]; r2, c2 = v_line[-1]
            v_top_closed = r1 == 0 or board[r1-1][c1] in ['m', 's']
            v_bottom_closed = r2 == 9 or board[r2+1][c2] in ['m', 's']
            v_closed = v_top_closed and v_bottom_closed
            
            # If horizontal is longer and NOT closed, suggest horizontal ends
            if len(h_line) > len(v_line) and not h_closed:
                r1, c1 = h_line[0]; r2, c2 = h_line[-1]
                targets = []
                if is_valid(r1, c1-1) and board[r1][c1-1] == '.':
                    targets.append(f"{col_map[c1-1]}{r1}")
                if is_valid(r2, c2+1) and board[r2][c2+1] == '.':
                    targets.append(f"{col_map[c2+1]}{r2}")
                if targets:
                    hints.append(f"!!! ACTIVE LINE from {col_map[c1]}{r1} to {col_map[c2]}{r2}. Target ends: {', '.join(targets)}.")
                continue
            
            # If vertical is longer and NOT closed, suggest vertical ends
            if len(v_line) > len(h_line) and not v_closed:
                r1, c1 = v_line[0]; r2, c2 = v_line[-1]
                targets = []
                if is_valid(r1-1, c1) and board[r1-1][c1] == '.':
                    targets.append(f"{col_map[c1]}{r1-1}")
                if is_valid(r2+1, c2) and board[r2+1][c2] == '.':
                    targets.append(f"{col_map[c2]}{r2+1}")
                if targets:
                    hints.append(f"!!! ACTIVE LINE from {col_map[c1]}{r1} to {col_map[c2]}{r2}. Target ends: {', '.join(targets)}.")
                continue
            
            # Both sequences equal length, or both closed - suggest all open neighbors
            for hit_r, hit_c in h_line:
                targets = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    tr, tc = hit_r+dr, hit_c+dc
                    if is_valid(tr, tc) and board[tr][tc] == '.':
                        targets.append(f"{col_map[tc]}{tr}")
                if targets:
                    hints.append(f"!!! ACTIVE HIT at {col_map[hit_c]}{hit_r}. Target neighbors: {', '.join(targets)}.")
        
        return "\n".join(hints) if hints else "No active hits. Proceed to Probability Search."


    # ========================================
    # PROBABILITY COMPUTATION
    # ========================================

    def find_active_hit_sequences(self, board: List[List[str]]) -> List[List[Tuple[int, int]]]:
        rows, cols = 10, 10
        visited = set()
        sequences = []
        for r in range(rows):
            for c in range(cols):
                if board[r][c] in ['h', 'a'] and (r, c) not in visited:
                    sequence = [(r, c)]
                    visited.add((r, c))
                    nc = c + 1
                    while nc < cols and board[r][nc] in ['h', 'a']:
                        sequence.append((r, nc)); visited.add((r, nc)); nc += 1
                    if len(sequence) == 1:
                        nr = r + 1
                        while nr < rows and board[nr][c] in ['h', 'a']:
                            sequence.append((nr, c)); visited.add((nr, c)); nr += 1
                    sequences.append(sequence)
        return sequences

    def compute_smart_llm_probabilities(self, board: List, remaining: List[int]) -> Dict[str, List]:
        grid = to_grid(board)
        active_sequences = self.find_active_hit_sequences(grid)
        R, C = 10, 10
        counts = [[0]*C for _ in range(R)]
        for L in remaining:
            if not isinstance(L, int) or L <= 0:
                continue
            placements = enumerate_valid_placements(grid, L)
            for p in placements:
                placement_coords = set(p)
                covers_active_hit = False
                for seq in active_sequences:
                    if any(coord in placement_coords for coord in seq):
                        if all(coord in placement_coords for coord in seq):
                            covers_active_hit = True
                            break
                weight = 1 if (not active_sequences or covers_active_hit) else 0
                for r, c in p:
                    counts[r][c] += weight
        for r in range(R):
            for c in range(C):
                if grid[r][c] != '.':
                    counts[r][c] = 0
        impossible = [[False]*C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if grid[r][c] != '.' or counts[r][c] == 0:
                    impossible[r][c] = True
        flat = [impossible[r][c] for r in range(R) for c in range(C)]
        return {"coverage": counts, "impossibleGrid": impossible, "impossibleFlat": flat, "active_sequences": active_sequences}

    # ========================================
    # FALLBACK LOGIC
    # ========================================

    def get_fallback_move(self, coverage_grid, board):
        self.used_fallback = True
        print(f"[{self.name}] Using fallback logic (LLM response failed)")
        candidates = []
        for r in range(10):
            for c in range(10):
                if board[r][c] == '.':
                    prob = coverage_grid[r][c]
                    neighbors = self.count_unknown_neighbors(board, r, c)
                    candidates.append((prob, neighbors, (r, c)))
        if not candidates:
            return (0, 0)
        candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    # ========================================
    # STATE UPDATE
    # ========================================

    def update_state(self, move, result, board):
        """Called after each move to update internal state."""
        r, c = move
        col_map = "ABCDEFGHIJ"
        coord = f"{col_map[c]}{r}"
        self.last_move = move
        self.update_agent_board(move, result)
        status_str = "HIT" if result.get('hit', False) else "Miss"
        if result.get('sunk_ship', None):
            status_str = "SUNK"
        self.move_history_log.append(f"{coord}: {status_str}")

    # ========================================
    # MAIN ORCHESTRATOR
    # ========================================

    async def select_move_async(self, board, remaining):
        """
        Entry point for move selection.
        Game passes in its board, but agent uses its own agent_board for deduction.
        """
        # Detect and process sinks
        ship_sank = len(remaining) < len(self.previous_remaining)
        if ship_sank and self.last_move is not None:
            sunk_size = self._compute_sunk_size_from_remaining(remaining, self.previous_remaining)
            if sunk_size is not None:
                # print(f"\n🚢 Ship of size {sunk_size} sank at move {self.last_move}")
                self.sunk_deductor(self.last_move, sunk_size)
                
        # CRITICAL: Attempt to resolve any old ambiguous patterns after board state changed.
        self.try_resolve_unsolved(remaining) 
        
        # Update internal state markers ('s' and 'a') based on deduction sets
        self.apply_deductions_to_agent_board() 

        self.previous_remaining = remaining.copy()
        self.validate_agent_board()
        display_board = self.create_display_board()
        features = self.compute_smart_llm_probabilities(display_board, remaining)
        turn = len(self.move_history_log) + 1
        self.display_board()
        return await self.select_move_logic(display_board, features, remaining, turn)

    async def select_move_logic(self, display_board, features, remaining_ships, turn):
        self.used_fallback = False
        self.api_call_count += 1  # Track attempt 
        tactical = self.get_tactical_hints(display_board)
        is_hunting = "ACTIVE" in tactical
        coverage_grid = features.get('coverage', np.zeros((10,10)))
        top_targets = self.get_top_probability_targets(coverage_grid, display_board, n=6)
        if is_hunting:
            self.last_prompt = f"""Battleship AI - HUNT MODE

BOARD:
{self.format_board(display_board)}
(. = unknown, h = hit, m = miss, s = sunk, a = ambiguous)

>>> PRIORITY TARGETS:
{tactical}

Ships remaining: {sorted(remaining_ships, reverse=True)}

Pick ONE target from the priority list above. Return JSON: {{"move": "X0", "reasoning": "..."}}
"""
        else:
            self.last_prompt = f"""Battleship AI - SEARCH MODE

BOARD:
{self.format_board(display_board)}
(. = unknown, h = hit, m = miss, s = sunk)

Best search targets (by probability): {top_targets}

Ships remaining: {sorted(remaining_ships, reverse=True)}

Pick a target. Return JSON: {{"move": "X0", "reasoning": "..."}}
"""
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Battleship AI. Return JSON only."},
                    {"role": "user", "content": self.last_prompt}
                ]
            )
            raw = completion.choices[0].message.content
            self.last_raw_json = raw
            self._track_usage(completion.usage)
            data = json_repair.loads(raw)
            if "move" not in data or "reasoning" not in data:
                raise ValueError("Missing 'move' or 'reasoning' in JSON")
            self.last_reasoning = data["reasoning"]
            move_str = data["move"].strip().upper()
            match = re.search(r"([A-J])\s*([0-9])", move_str)
            if match:
                col_char = match.group(1); row_char = match.group(2)
                c = "ABCDEFGHIJ".index(col_char); r = int(row_char)
                if 0 <= r < 10 and 0 <= c < 10:
                    if self.agent_board[r][c] in ['.', '_']:
                        self.api_call_count += 1  # Track attempt 
                        return (r, c)
                    else:
                        print(f"[{self.name} Warning] Cell already played: {move_str}")
                        error_msg = f"Cell already played: {move_str}"
                        print(f"[{self.name} Warning] {error_msg}")
                        self.log_api_failure(ValueError, error_msg, {'move_str': move_str, 'cell_state': self.agent_board[r][c]})
            else:
                error_msg = f"Invalid move format: {move_str}"
                print(f"[{self.name} Warning] {error_msg}")
                self.log_api_failure(ValueError, error_msg, {'move_str': move_str, 'raw_response': raw[:100]})

        except json.JSONDecodeError as e:
            print(f"[{self.name} JSON Error] {e}")
            self.log_api_failure(type(e), str(e), {'raw_response_preview': self.last_raw_json[:200] if self.last_raw_json else None})
        except ValueError as e:
            print(f"[{self.name} Validation Error] {e}")
            self.log_api_failure(type(e), str(e), {'parsed_data': data if 'data' in locals() else None})
        except Exception as e:
            print(f"[{self.name} Error] {e}")
            self.log_api_failure(type(e), str(e), {'exception_class': type(e).__name__})

        return self.get_fallback_move(coverage_grid, self.agent_board)

    def display_board(self):
        """Prints the agent's single internal board (self.agent_board)."""
        header = "   A B C D E F G H I J"
        print(header)
        
        # Iterate over the rows of the agent's board
        for r in range(10):
            # Access the row list directly: self.agent_board[r]
            row_str = f"{r}  " + " ".join(self.agent_board[r])
            print(row_str)
            
        print()  

    def reset_state(self):
        """Resets the agent's internal state for a new game."""
        print("--- Agent State Reset Initiated ---")
        
        # 1. Reset Board and Ship Data
        self.agent_board = [['.' for _ in range(10)] for _ in range(10)]
        self.previous_remaining = self.initial_ships.copy()
        
        # 2. Clear Tracking Sets
        self.sunk_set = set()
        self.ambig_set = set()
        self.unsolved = set()
        
        # 3. Clear History and Logs
        self.move_history_log = []
        self.last_move = None
        self.last_reasoning = None
        self.last_prompt = None
        self.last_raw_json = None
        self.used_fallback = False

        # Reset API failure tracking
        self.api_failures = []
        self.api_call_count = 0
        self.api_success_count = 0
        print("--- Agent State Reset Complete ---")
