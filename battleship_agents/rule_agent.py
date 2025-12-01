import random

class RuleBasedAgent:
    """
    A Battleship agent that uses hunt/target logic:
    - Hunt mode: random shots on unhit cells
    - Target mode: focus around existing hits
    - Axis detection: align shots if multiple hits suggest orientation
    """

    def __init__(self, name="RuleBasedAgent"):
        self.name = name
        self.hit_queue = []   # cells with hits not yet sunk
        self.shot_queue = []  # candidate moves near hits
        self.agent_type = "heuristic"

    def near_moves(self, cell, board):
        """Return orthogonal neighbors of a cell that are still valid."""
        r, c = cell
        moves = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == ".":
                moves.append((nr,nc))
        return moves

    def detect_axis(self, hits):
        """Detect orientation axis if multiple hits suggest a line."""
        if len(hits) < 2:
            return None
        rows = [r for r,c in hits]
        cols = [c for r,c in hits]
        if len(set(rows)) == 1:   # same row → horizontal
            return ("row", rows[0])
        if len(set(cols)) == 1:   # same col → vertical
            return ("col", cols[0])
        return None

    def select_move(self, board, features=None, remaining=None, turn=0):
        # Keep only hits that are not marked sunk
        self.hit_queue = [(r,c) for (r,c) in self.hit_queue if board[r][c] == "h"]

        # Build shot_queue from neighbors of hits
        self.shot_queue = []
        for hit in self.hit_queue:
            self.shot_queue.extend(self.near_moves(hit, board))

        # Axis alignment
        axis = self.detect_axis(self.hit_queue)
        if axis and self.shot_queue:
            if axis[0] == "row":
                self.shot_queue = [(r,c) for (r,c) in self.shot_queue if r == axis[1]]
            elif axis[0] == "col":
                self.shot_queue = [(r,c) for (r,c) in self.shot_queue if c == axis[1]]

        # Target mode
        if self.hit_queue and self.shot_queue:
            move = self.shot_queue[0]
        else:
            # Hunt mode: random unhit cell
            legal = [(r,c) for r in range(10) for c in range(10) if board[r][c] == "."]
            move = random.choice(legal) if legal else (0,0)

        return move

    def update_after_shot(self, move, board, is_hit, is_sunk=False):
        """Update queues after a shot result (environment tells us hit/sunk)."""
        r,c = move
        if is_hit:
            self.hit_queue.append((r,c))
            self.shot_queue.extend(self.near_moves((r,c), board))
            self.shot_queue = [(rr,cc) for (rr,cc) in self.shot_queue if board[rr][cc] == "."]
            if is_sunk:
                # Clear queues once environment confirms sunk
                self.hit_queue = []
                self.shot_queue = []
