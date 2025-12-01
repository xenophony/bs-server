import numpy as np
import random
import joblib

class QAgentWrapper:
    def __init__(self, q_agent, name="QLearningAgent"):
        self.q_agent = q_agent
        self.name = name
        self.agent_type = "rl"

    def select_move(self, board, features, remaining=None, turn=0):
        # Build observation in the format QLearningAgent expects
        obs = {
            "board": board,
            "remaining_ships": remaining
        }
        action = self.q_agent.act(obs)   # flat index 0–99
        return divmod(action, 10)        # convert to (r, c)


class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999995, min_epsilon=0.1):
        # Q-table indexed by: [action, adj_hits, adj_misses, is_empty, largest_remaining_ship]
        self.q_table = np.zeros((100, 3, 3, 2, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_state_features(self, obs, action):
        """Extract spatial features around the action cell."""
        board = np.array(obs['board'])   # ✅ ensure NumPy array
        remaining_ships = obs['remaining_ships']

        r, c = divmod(action, 10)
        flat = board.flatten()

        # Count orthogonally adjacent hits and misses
        adjacent_hits = 0
        adjacent_misses = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                idx = nr * 10 + nc
                if flat[idx] == 'h':
                    adjacent_hits += 1
                elif flat[idx] == 'm':
                    adjacent_misses += 1

        adjacent_hits = min(adjacent_hits, 2)
        adjacent_misses = min(adjacent_misses, 2)
        is_empty = 1 if flat[action] == "." else 0

        if remaining_ships:
            largest = max(remaining_ships)
            largest_remaining = min(largest - 2, 3)
        else:
            largest_remaining = 0

        return (adjacent_hits, adjacent_misses, is_empty, largest_remaining)

    def act(self, obs):
        board = np.array(obs['board'])   # ✅ ensure NumPy array
        flat = board.flatten()
        legal_actions = [i for i in range(100) if flat[i] == "."]
        if not legal_actions:
            return random.randint(0, 99)

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        q_values = []
        for a in legal_actions:
            h, m, e, l = self.get_state_features(obs, a)
            q_values.append(self.q_table[a, h, m, e, l])

        return legal_actions[int(np.argmax(q_values))]

    def learn(self, obs, action, reward, next_obs, done):
        h, m, e, l = self.get_state_features(obs, action)

        board = np.array(next_obs['board'])   # ✅ ensure NumPy array
        flat = board.flatten()
        legal_actions = [i for i in range(100) if flat[i] == "."]

        if legal_actions and not done:
            next_q_values = []
            for a in legal_actions:
                nh, nm, ne, nl = self.get_state_features(next_obs, a)
                next_q_values.append(self.q_table[a, nh, nm, ne, nl])
            best_next_q = max(next_q_values)
        else:
            best_next_q = 0.0

        td_target = reward + self.gamma * best_next_q * (not done)
        td_error = td_target - self.q_table[action, h, m, e, l]
        self.q_table[action, h, m, e, l] += self.alpha * td_error

    def save(self, path="q_table.pkl"):
        joblib.dump(self.q_table, path)

    def load(self, path="q_table.pkl"):
        self.q_table = joblib.load(path)
