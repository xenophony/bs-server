# battleship_agents/probability_agent.py

from typing import List, Tuple
from battleship_agents.board_utils import compute_impossible_mask

class ProbabilityHeatmapAgent:
    """
    Agent that uses probability density heatmap to select optimal moves.
    Calculates how many ship configurations can fit at each cell.
    """
    
    def __init__(self, name="ProbabilityAgent"):
        self.name = name
        self.agent_type = "heuristic"
    
    def compute_heatmap(self, board: List[List[str]], remaining: List[int]) -> List[List[int]]:
        """
        Calculate probability heatmap based on valid ship placements.
        Returns 10x10 grid where each cell = number of possible ship placements covering it.
        """
        heatmap = [[0] * 10 for _ in range(10)]
        
        for ship_size in remaining:
            # Check all possible horizontal placements
            for r in range(10):
                for c in range(10 - ship_size + 1):
                    if self._can_place_horizontal(board, r, c, ship_size):
                        for offset in range(ship_size):
                            heatmap[r][c + offset] += 1
            
            # Check all possible vertical placements
            for r in range(10 - ship_size + 1):
                for c in range(10):
                    if self._can_place_vertical(board, r, c, ship_size):
                        for offset in range(ship_size):
                            heatmap[r + offset][c] += 1
        
        return heatmap
    
    def _can_place_horizontal(self, board: List[List[str]], r: int, c: int, size: int) -> bool:
        """Check if ship can be placed horizontally starting at (r,c)"""
        for offset in range(size):
            cell = board[r][c + offset]
            if cell in ['m', 'h', 's']:  # Miss, hit, or sunk
                return False
        return True
    
    def _can_place_vertical(self, board: List[List[str]], r: int, c: int, size: int) -> bool:
        """Check if ship can be placed vertically starting at (r,c)"""
        for offset in range(size):
            cell = board[r + offset][c]
            if cell in ['m', 'h', 's']:
                return False
        return True
    
    def select_move(self, board: List[List[str]], features=None, remaining=None, turn=None) -> Tuple[int, int]:
        """
        Select move with highest probability from heatmap.
        """
        if remaining is None:
            remaining = [5, 4, 3, 3, 2]  # Default ship sizes
        
        # Compute probability heatmap
        heatmap = self.compute_heatmap(board, remaining)
        
        # Find cell with highest probability that hasn't been shot
        max_prob = -1
        best_move = None
        
        for r in range(10):
            for c in range(10):
                cell = board[r][c]
                prob = heatmap[r][c]
                
                # Only consider unshot cells
                if cell == '.' and prob > max_prob:
                    max_prob = prob
                    best_move = (r, c)
        
        # Fallback to first available cell if no valid move found
        if best_move is None:
            for r in range(10):
                for c in range(10):
                    if board[r][c] == '.':
                        return (r, c)
        
        return best_move if best_move else (0, 0)


class EnhancedProbabilityAgent(ProbabilityHeatmapAgent):
    """
    Enhanced version with hunt mode after hits.
    """
    
    def __init__(self, name="EnhancedProbabilityAgent"):
        super().__init__(name)
        self.active_hits = []  # Track unsunk hits
    
    def update_state(self, move: Tuple[int, int], result: dict, board: List[List[str]]):
        """Update internal state after each move"""
        if result['hit'] and not result['sunk_ship']:
            self.active_hits.append(move)
        elif result['sunk_ship']:
            # Remove all hits from the sunk ship
            self.active_hits = [h for h in self.active_hits 
                               if h not in result['sunk_ship'].cells]
    
    def get_adjacent_cells(self, r: int, c: int, board: List[List[str]]) -> List[Tuple[int, int]]:
        """Get valid adjacent cells (not already shot)"""
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == '.':
                adjacent.append((nr, nc))
        return adjacent
    
    def select_move(self, board: List[List[str]], features=None, remaining=None, turn=None) -> Tuple[int, int]:
        """
        Hunt mode: Target adjacent cells to active hits
        Search mode: Use probability heatmap
        """
        if remaining is None:
            remaining = [5, 4, 3, 3, 2]
        
        # HUNT MODE: If we have active hits, target adjacent cells
        if self.active_hits:
            # Compute heatmap for adjacent cells
            heatmap = self.compute_heatmap(board, remaining)
            
            best_prob = -1
            best_move = None
            
            for hit_r, hit_c in self.active_hits:
                for adj_r, adj_c in self.get_adjacent_cells(hit_r, hit_c, board):
                    prob = heatmap[adj_r][adj_c]
                    if prob > best_prob:
                        best_prob = prob
                        best_move = (adj_r, adj_c)
            
            if best_move:
                return best_move
        
        # SEARCH MODE: Use standard heatmap
        return super().select_move(board, features, remaining, turn)
