import random
import random

class Ship:
    def __init__(self, size, cells):
        self.size = size
        self.cells = set(cells)
        self.hits = set()

    def register_hit(self, r, c):
        if (r, c) in self.cells:
            self.hits.add((r, c))
            return True
        return False

    def is_sunk(self):
        return len(self.hits) == self.size

def empty_board():
    return [["."] * 10 for _ in range(10)]

def place_ships(board, ships):
    ship_objects = []
    for size in ships:
        placed = False
        while not placed:
            r, c = random.randint(0, 9), random.randint(0, 9)
            orientation = random.choice(["H", "V"])
            if orientation == "H" and c + size <= 10:
                if all(board[r][cc] == "." for cc in range(c, c+size)):
                    cells = [(r, cc) for cc in range(c, c+size)]
                    for rr, cc in cells:
                        board[rr][cc] = "_"  # hidden ship marker
                    ship_objects.append(Ship(size, cells))
                    placed = True
            elif orientation == "V" and r + size <= 10:
                if all(board[rr][c] == "." for rr in range(r, r+size)):
                    cells = [(rr, c) for rr in range(r, r+size)]
                    for rr, cc in cells:
                        board[rr][cc] = "_"
                    ship_objects.append(Ship(size, cells))
                    placed = True
    return board, ship_objects



# def empty_board():
#     return [["."] * 10 for _ in range(10)]

# def place_ships(board, ships):
#     """
#     Randomly place ships on the board.
#     Ships is a list of lengths, e.g. [5,4,3,3,2].
#     """
#     for size in ships:
#         placed = False
#         while not placed:
#             r, c = random.randint(0, 9), random.randint(0, 9)
#             orientation = random.choice(["H", "V"])
#             if orientation == "H" and c + size <= 10:
#                 if all(board[r][cc] == "." for cc in range(c, c+size)):
#                     for cc in range(c, c+size):
#                         board[r][cc] = "h"  # mark ship cells as hidden hits
#                     placed = True
#             elif orientation == "V" and r + size <= 10:
#                 if all(board[rr][c] == "." for rr in range(r, r+size)):
#                     for rr in range(r, r+size):
#                         board[rr][c] = "h"
#                     placed = True
#     return board
