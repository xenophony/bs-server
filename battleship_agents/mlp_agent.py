import numpy as np
import joblib
import tensorflow as tf
from battleship_agents.base_agent import BaseAgent
from features import make_record

class MLPAgent(BaseAgent):
    def __init__(self, model_path, scaler_path, features_path, name="MLPAgent"):
        super().__init__(name)
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.agent_type = "ml"
        with open(features_path) as f:
            self.feature_list = [line.strip() for line in f]

    def select_move(self, board, features, remaining=None, turn=0):
        # Build all feature vectors at once for batch prediction
        records = []
        valid_positions = []
        
        for r in range(10):
            for c in range(10):
                if board[r][c] in [".", "_"]:  # Only evaluate legal moves
                    idx = r * 10 + c
                    record = make_record(board, remaining, turn, idx)
                    x = [record[feat] for feat in self.feature_list]
                    records.append(x)
                    valid_positions.append((r, c))
        
        if not records:
            return (0, 0)  # Fallback
        
        # Batch prediction - much faster!
        X = np.array(records)
        X_scaled = self.scaler.transform(X)
        scores = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Find best move
        best_idx = np.argmax(scores)
        return valid_positions[best_idx]

