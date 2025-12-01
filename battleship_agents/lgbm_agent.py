import numpy as np
import joblib
import tensorflow as tf
from battleship_agents.base_agent import BaseAgent
from features import make_record


class LGBMAgent(BaseAgent):
    def __init__(self, model_path, features_path, name="LGBMAgent"):
        super().__init__(name)
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=model_path)
        self.agent_type = "ml"
        with open(features_path) as f:
            self.feature_list = [line.strip() for line in f]

    def select_move(self, board, features, remaining=None, turn=0):
        records, valid_positions = [], []
        for r in range(10):
            for c in range(10):
                if board[r][c] in [".", "_"]:
                    idx = r * 10 + c
                    record = make_record(board, remaining, turn, idx)
                    x = [record[feat] for feat in self.feature_list]
                    records.append(x)
                    valid_positions.append((r, c))

        if not records:
            return (0, 0)

        X = np.array(records)
        scores = self.model.predict(X)  # LightGBM booster returns probabilities
        best_idx = np.argmax(scores)
        return valid_positions[best_idx]