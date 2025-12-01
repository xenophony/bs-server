import json
import os
import re
import random
import json_repair
from collections import Counter
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class FrontierAgent:
    """
    Adapted FineTunedAgent for Frontier Models (GPT-4o, Claude 3.5, Gemini).
    
    - Maintains internal board state and ship deduction logic.
    - Sends clean JSON-formatted Chat messages (System/User).
    - Uses standard chat completions with JSON mode for broad compatibility.
    - Removes probability hints to force raw reasoning.
    """
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
        
        # Tracking sets for Deduction
        self.sunk_set = set()          # Confirmed sunk ship coordinates
        self.ambig_set = set()         # Ambiguous hit coordinates
        self.unsolved = set()          # {((r,c), size), ...}
        
        # Economics
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # API Failure Tracking
        self.api_failures = [] 
        self.api_call_count = 0 
        self.api_success_count = 0 

        if client:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ========================================
    # UTILITIES
    # ========================================

    def _track_usage(self, usage_obj):
        if usage_obj:
            self.token_usage["prompt_tokens"] += usage_obj.prompt_tokens
            self.token_usage["completion_tokens"] += usage_obj.completion_tokens
            self.token_usage["total_tokens"] += usage_obj.total_tokens

    def log_api_failure(self, error_type, error_message, context=None):
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'turn': len(self.move_history_log) + 1,
            'error_type': str(error_type),
            'error_message': str(error_message),
            'model': self.model,
            'last_prompt_preview': self.last_prompt[:200] if self.last_prompt else None,
            'last_raw_response': self.last_raw_json,
            'context': context or {}
        }
        self.api_failures.append(failure_record)

    def get_failure_metadata(self):
        return {
            'total_api_calls': self.api_call_count,
            'successful_calls': self.api_success_count,
            'failed_calls': len(self.api_failures),
            'failures': self.api_failures,
            'fallback_usage': self.used_fallback
        }

    # ========================================
    # BOARD MANAGEMENT
    # ========================================
    
    def update_agent_board(self, move, result):
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
        """Returns a copy of the board for display/logic."""
        return [row[:] for row in self.agent_board]

    def validate_agent_board(self):
        allowed = {'.', 'm', 'h', 'a', 's'}
        for r, row in enumerate(self.agent_board):
            for c, cell in enumerate(row):
                if cell not in allowed:
                    raise ValueError(f"Invalid cell '{cell}' at {(r,c)}")

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
        board = self.agent_board

        def get_direction_sequence(dr, dc):
            seq = [last_move]
            max_seq = [last_move]
            k = 1
            while True:
                nr, nc = last_move[0] + (dr * k), last_move[1] + (dc * k)
                if 0 <= nr <= 9 and 0 <= nc <= 9 and board[nr][nc] in ['h', 'a']:
                    seq.append((nr, nc))
                    if k < sunk_ship:
                        max_seq.append((nr, nc))
                    k += 1
                else:
                    break
            return seq, max_seq

        def check_end(dr, dc):
            nr, nc = last_move[0] + dr, last_move[1] + dc
            return not (0 <= nr <= 9 and 0 <= nc <= 9 and board[nr][nc] in ['h', 'a'])

        up_seq, max_up = get_direction_sequence(1, 0)
        down_seq, max_down = get_direction_sequence(-1, 0)
        right_seq, max_right = get_direction_sequence(0, 1)
        left_seq, max_left = get_direction_sequence(0, -1)
        
        # Combine sequences (current cell is in both, so -1 to length calculation)
        len_vert = len(up_seq) + len(down_seq) - 1
        len_horz = len(right_seq) + len(left_seq) - 1

        is_vert_end = check_end(1, 0) or check_end(-1, 0)
        is_horz_end = check_end(0, 1) or check_end(0, -1)

        # Logic to determine if solved or ambiguous
        if len_vert == sunk_ship and len_horz < sunk_ship:
            return list(set(up_seq + down_seq)), []
        if len_horz == sunk_ship and len_vert < sunk_ship:
            return list(set(right_seq + left_seq)), []

        # Ambiguous cases
        ambig_candidates = []
        if len_vert >= sunk_ship:
            ambig_candidates.extend(max_up + max_down)
        if len_horz >= sunk_ship:
            ambig_candidates.extend(max_right + max_left)
            
        return [], list(set(ambig_candidates))

    def update_board_sunk_ambig(self, sunk, ambig):
        self.sunk_set.update(sunk)
        self.ambig_set.update(ambig)
        self.ambig_set.difference_update(self.sunk_set)

    def sunk_deductor(self, last_move, sunk_size):
        sunk, ambig = self.determine_sunk_ship(sunk_size, last_move)
        self.update_board_sunk_ambig(sunk, ambig)
        self.apply_deductions_to_agent_board()
        
        if not sunk and ambig:
            self.unsolved.add((last_move, sunk_size))
        else:
            self.unsolved.discard((last_move, sunk_size))
            self.try_resolve_unsolved(self.previous_remaining)

    def try_resolve_unsolved(self, remaining):
        if not self.unsolved:
            return
        
        new_unsolved = set()
        for (coord, size) in list(self.unsolved):
            if coord in self.sunk_set:
                continue

            sunk, ambig = self.determine_sunk_ship(size, coord)
            if sunk:
                self.sunk_set.update(sunk)
                self.ambig_set.difference_update(sunk)
            elif ambig:
                new_unsolved.add((coord, size))
        
        self.unsolved = new_unsolved

    # ========================================
    # PROMPT CONSTRUCTION
    # ========================================

    def _format_board_string(self):
        """Formats board exactly as seen in training data."""
        header = "   A B C D E F G H I J"
        rows = [header]
        for r in range(10):
            row_content = " ".join(self.agent_board[r])
            rows.append(f"{r}  {row_content}")
        return "\n".join(rows)

    def _build_chat_prompt(self, remaining):
        board_str = self._format_board_string()
        
        prompt = f"""You are an expert Battleship player.
Objective: Sink all hidden ships.

BOARD STATE:
{board_str}

LEGEND:
. = Unknown
h = Hit
m = Miss
s = Sunk (Confirmed)
a = Ambiguous (Hit, but ship undefined)

SHIPS REMAINING: {sorted(remaining, reverse=True)}

INSTRUCTIONS:
1. Analyze the board patterns.
2. If there are active hits ('h' or 'a'), prioritize HUNTING them to sink the ship.
3. If not, SEARCH for probable locations based on the remaining ship sizes.
4. Output your move and reasoning in the enforced JSON format.

JSON RESPONSE FORMAT EXAMPLE:
{{
  "move": "A1", 
  "reasoning": "Targeting A1 because it is adjacent to a hit and follows a diagonal pattern."
}}
"""
        return prompt

    # ========================================
    # MAIN LOGIC
    # ========================================

    def get_fallback_move(self, board):
        """Simple random fallback if LLM fails."""
        self.used_fallback = True
        candidates = []
        for r in range(10):
            for c in range(10):
                if board[r][c] == '.':
                    candidates.append((r, c))
        
        if candidates:
            return random.choice(candidates)
        return (0, 0)

    def update_state(self, move, result, board):
        """Called by the game runner after a move."""
        r, c = move
        self.last_move = move
        self.update_agent_board(move, result)
        
        status_str = "HIT" if result.get('hit', False) else "Miss"
        if result.get('sunk_ship', None):
            status_str = "SUNK"
        
        col_map = "ABCDEFGHIJ"
        self.move_history_log.append(f"{col_map[c]}{r}: {status_str}")

    async def select_move_async(self, board, remaining):
        """Main entry point for the Agent."""
        
        # 1. Deduce Sunk Ships (Smart Logic)
        ship_sank = len(remaining) < len(self.previous_remaining)
        if ship_sank and self.last_move is not None:
            sunk_size = self._compute_sunk_size_from_remaining(remaining, self.previous_remaining)
            if sunk_size:
                self.sunk_deductor(self.last_move, sunk_size)

        # 2. Resolve Ambiguities & Update State
        self.try_resolve_unsolved(remaining)
        self.apply_deductions_to_agent_board()
        self.previous_remaining = remaining.copy()
        self.validate_agent_board()

        # 3. Generate Prompt (Frontier Style)
        prompt = self._build_chat_prompt(remaining)
        self.last_prompt = prompt

        # 4. API Call (Standard Chat Completion)
        try:
            # Use standard chat.completions.create with json_object mode
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Battleship AI. You must always respond directly in JSON and must include move and reasoning, do not add any extra markup or markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150,
                response_format={"type": "json_object"} 
            )
            
            if hasattr(response, 'usage'):
                self._track_usage(response.usage)
            
            raw_content = response.choices[0].message.content
            self.last_raw_json = raw_content
            
            # --- IMPROVED PARSING LOGIC ---
            # Extract JSON substring using regex to handle extra text/markdown
            json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
            if json_match:
                clean_json = json_match.group(1)
            else:
                clean_json = raw_content # Fallback to raw if regex fails
            
            # Use json_repair for robustness
            try:
                data = json_repair.loads(clean_json)
            except Exception:
                data = json.loads(clean_json) # Last ditch effort

            # Check for double encoding
            if isinstance(data, str):
                try:
                    data = json_repair.loads(data)
                except Exception:
                    pass

            if not isinstance(data, dict):
                 raise ValueError(f"Response is not a valid JSON object: {raw_content[:50]}...")

            self.last_reasoning = data.get("reasoning", "")
            move_str = data.get("move", "").strip().upper()
            
            # Parse Coordinates
            if len(move_str) >= 2:
                col_char = move_str[0]
                row_char = move_str[1:]
                if col_char in "ABCDEFGHIJ" and row_char.isdigit():
                    c = "ABCDEFGHIJ".index(col_char)
                    r = int(row_char)
                    
                    if 0 <= r < 10 and 0 <= c < 10:
                        if self.agent_board[r][c] in ['.', 'a', 'h', 's']: 
                            self.api_success_count += 1
                            return (r, c)
            
            print(f"[{self.name}] Invalid move: {move_str}")
            self.log_api_failure(ValueError, f"Invalid move string: {move_str}")

        except Exception as e:
            print(f"[{self.name}] API Error: {e}")
            self.log_api_failure(type(e), str(e))

        return self.get_fallback_move(self.agent_board)

    def display_board(self):
        # Optional: Print board for debugging
        print(self._format_board_string())

    def reset_state(self):
        self.agent_board = [['.' for _ in range(10)] for _ in range(10)]
        self.previous_remaining = self.initial_ships.copy()
        self.sunk_set = set()
        self.ambig_set = set()
        self.unsolved = set()
        self.move_history_log = []
        self.last_move = None
        self.last_reasoning = None
        self.last_prompt = None
        self.last_raw_json = None
        self.used_fallback = False
        self.api_failures = []