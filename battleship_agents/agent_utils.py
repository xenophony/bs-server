class RLAgentWrapper:
    def __init__(self, q_agent, name="QLearningAgent"):
        self.q_agent = q_agent
        self.name = name

    def select_move(self, board, features, remaining=None, turn=0):
        # Build observation in the format QLearningAgent expects
        obs = {
            "board": board,
            "remaining_ships": remaining
        }
        action = self.q_agent.act(obs)   # flat index 0–99
        return divmod(action, 10)        # convert to (r, c)
    



