import numpy as np
import joblib
import tensorflow as tf
from battleship_agents.base_agent import BaseAgent
from features import make_record


class SklearnAgent(BaseAgent):
    def __init__(self, model_path, features_path, scaler_path=None, name="SklearnAgent"):
        super().__init__(name)
        self.model = joblib.load(model_path)
        self.agent_type = "ml"
        self.scaler = joblib.load(scaler_path) if scaler_path else None
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
        if self.scaler:
            X = self.scaler.transform(X)

        # scikit-learn models: predict_proba
        scores = self.model.predict_proba(X)[:, 1]
        best_idx = np.argmax(scores)
        return valid_positions[best_idx]