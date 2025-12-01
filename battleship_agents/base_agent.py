class BaseAgent:
    def __init__(self, name): self.name = name
    def select_move(self, board, features): raise NotImplementedError
