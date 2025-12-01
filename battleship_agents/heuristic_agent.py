from battleship_agents.base_agent import BaseAgent
import random

class HeuristicAgent(BaseAgent):
    def __init__(self, name="HeuristicAgent"):
        super().__init__(name)
        self.agent_type = "heuristic"

    def select_move(self, board, features, remaining=None, turn=0):
        coverage = features["coverage"]
        impossible = features["impossibleGrid"]

        best_val, best_move = -1, None
        for r in range(10):
            for c in range(10):
                # Treat "." (unknown water) and "_" (hidden ship) as legal targets
                if board[r][c] in [".", "_"] and not impossible[r][c]:
                    if coverage[r][c] > best_val:
                        best_val = coverage[r][c]
                        best_move = (r, c)

        # Fallback: pick the first legal cell if no coverage candidate found
        if best_move is None:
            legal = [(r, c) for r in range(10) for c in range(10) if board[r][c] in [".", "_"]]
            if legal:
                return legal[0]
            else:
                # Absolute fallback: random cell (shouldn’t happen unless board is exhausted)
                return (0, 0)

        return best_move
