import sqlite3
import time
import os
from typing import List, Dict, Optional

# Use a persistent path for Railway volumes, or fallback to local file
DB_FILE = os.getenv("LEADERBOARD_DB_PATH", "leaderboard.db")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    """Create the leaderboard table if it doesn't exist."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Updated Schema: Added 'mode' and 'player_type'
    # Note: If you already have a DB, you might need to delete it or migrate it manually.
    c.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            turns INTEGER NOT NULL,
            difficulty TEXT NOT NULL,  -- e.g. "smart-prob", "frontier-claude"
            mode TEXT NOT NULL,        -- "classic" or "race"
            player_type TEXT NOT NULL, -- "human" or "agent"
            timestamp REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_score(player_name: str, turns: int, difficulty: str, mode: str, player_type: str):
    """Insert a new high score."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        '''INSERT INTO leaderboard 
           (player_name, turns, difficulty, mode, player_type, timestamp) 
           VALUES (?, ?, ?, ?, ?, ?)''',
        (player_name, turns, difficulty, mode, player_type, time.time())
    )
    conn.commit()
    conn.close()

def get_top_scores(limit: int = 10, mode: str = "classic", player_type: str = "human") -> List[Dict]:
    """
    Get the top N scores, filtered by mode and player type.
    Example: get_top_scores(10, "race", "agent") -> Top fastest agents in Race Mode.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT player_name, turns, difficulty, timestamp 
        FROM leaderboard 
        WHERE mode = ? AND player_type = ?
        ORDER BY turns ASC 
        LIMIT ?
    ''', (mode, player_type, limit))
        
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]