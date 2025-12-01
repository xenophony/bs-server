import numpy as np
import random
import joblib

class SARSAAgent:
    def __init__(self, alpha=0.3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999995, min_epsilon=0.1):
        # Q-table indexed by: [action, adj_hits, adj_misses, is_empty, largest_remaining_ship]
        self.q_table = np.zeros((100, 3, 3, 2, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.agent_type = "rl"

    def get_state_features(self, obs, action):
        board = np.array(obs['board'])
        remaining_ships = obs['remaining_ships']
        r, c = divmod(action, 10)
        flat = board.flatten()

        # Adjacent hits/misses
        adjacent_hits = 0
        adjacent_misses = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                idx = nr*10 + nc
                if flat[idx] == 'h':
                    adjacent_hits += 1
                elif flat[idx] == 'm':
                    adjacent_misses += 1
        adjacent_hits = min(adjacent_hits, 2)
        adjacent_misses = min(adjacent_misses, 2)
        is_empty = 1 if flat[action] == "." else 0
        largest_remaining = min(max(remaining_ships) - 2, 3) if remaining_ships else 0
        return (adjacent_hits, adjacent_misses, is_empty, largest_remaining)

    def act(self, obs):
        board = np.array(obs['board'])
        flat = board.flatten()
        legal_actions = [i for i in range(100) if flat[i] == "."]
        if not legal_actions:
            return random.randint(0, 99)
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        q_values = [self.q_table[a, *self.get_state_features(obs, a)] for a in legal_actions]
        return legal_actions[int(np.argmax(q_values))]

    def learn(self, obs, action, reward, next_obs, next_action, done):
        h, m, e, l = self.get_state_features(obs, action)
        nh, nm, ne, nl = self.get_state_features(next_obs, next_action)
        td_target = reward + self.gamma * self.q_table[next_action, nh, nm, ne, nl] * (not done)
        td_error = td_target - self.q_table[action, h, m, e, l]
        self.q_table[action, h, m, e, l] += self.alpha * td_error
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path="sarsa_table_featurese.pkl"):
        joblib.dump(self.q_table, path)

    def load(self, path="sarsa_table_features.pkl"):
        self.q_table = joblib.load(path)
