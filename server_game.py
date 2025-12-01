import uuid
import time
import asyncio
import os
import joblib
import pickle
import json
import copy 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- IMPORT THIS
from pydantic import BaseModel
from typing import Optional, Dict, List
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- AGENT IMPORTS ---
from battleship_agents.smart_llm_agent_v2 import SmartLLMAgentV2
from battleship_agents.smart_probability_agent_v2 import SmartProbabilityAgent
from battleship_agents.sklearn_agent import SklearnAgent
from battleship_agents.lgbm_agent import LGBMAgent
from battleship_agents.mlp_agent import MLPAgent
from battleship_agents.q_agent import QLearningAgent
from battleship_agents.sarsa_agent import SARSAAgent
from battleship_agents.agent_utils import RLAgentWrapper
from battleship_agents.rule_agent import RuleBasedAgent
from battleship_agents.heuristic_agent import HeuristicAgent
from battleship_agents.board_setup import place_ships, empty_board, Ship
from server_leaderboard import init_db, add_score, get_top_scores

# Ensure 'features' module is available for ML agents
try:
    from features import make_record
except ImportError:
    print("Warning: features.py not found in root. ML agents will likely fail.")

load_dotenv()

app = FastAPI()

# --- CORS CONFIGURATION (CRITICAL FOR REACT CLIENT) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g. localhost:3001, railway app)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- STATE MANAGEMENT ---
games: Dict[str, dict] = {}

# --- API CLIENT CONFIGURATION ---
# Only Local Client is needed now
local_client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",  # ✅ CORRECT
    api_key=os.getenv("TOGETHER_API_KEY")
)


# --- REQUEST MODELS ---
class ShipPlacement(BaseModel):
    name: str 
    size: int
    coords: List[List[int]]

class GameStartRequest(BaseModel):
    difficulty: str = "smart-prob"
    mode: str = "classic" # "classic" or "race"
    player_ships: Optional[List[ShipPlacement]] = None

class PlayerTurnRequest(BaseModel):
    game_id: str
    player_shot: str 
    ai_last_shot_result: Optional[str] = None 
    player_ships_remaining: List[int] 

class ScoreSubmission(BaseModel):
    game_id: str
    player_name: str

# --- HELPER: INITIALIZE AGENT ---
def get_agent_by_difficulty(difficulty: str):
    """Factory to create the requested agent based on difficulty string."""
    
    # === HEURISTIC AGENTS ===
    if difficulty == "smart-prob":
        agent = SmartProbabilityAgent(name="Smart-Prob-Algorithm")
        agent.agent_type = "heuristic"
        return agent
    
    elif difficulty == "rule-based":
        agent = RuleBasedAgent()
        agent.agent_type = "heuristic"
        return agent
    
    elif difficulty == "heuristic":
        agent = HeuristicAgent()
        agent.agent_type = "heuristic"
        return agent
    
    # === MACHINE LEARNING AGENTS ===
    elif difficulty == "ml-logistic":
        if os.path.exists("ml_models/lr.joblib"):
            agent = SklearnAgent(
                model_path="ml_models/lr.joblib",
                features_path="ml_models/featureslist.txt",
                name="LogisticRegression"
            )
            agent.agent_type = "ml"
            return agent
        return SmartProbabilityAgent(name="Fallback-Prob")

    elif difficulty == "ml-hgb":
        if os.path.exists("ml_models/hgb.joblib"):
            agent = SklearnAgent(
                model_path="ml_models/hgb.joblib",
                features_path="ml_models/featureslist.txt",
                name="HitGradientBoosted"
            )
            agent.agent_type = "ml"
            return agent
        return SmartProbabilityAgent(name="Fallback-Prob")

    elif difficulty == "ml-lgbm":
        if os.path.exists("ml_models/lgbm.txt"):
            agent = LGBMAgent(
                model_path="ml_models/lgbm.txt",
                features_path="ml_models/featureslist.txt",
                name="LightGBM"
            )
            agent.agent_type = "ml"
            return agent
        return SmartProbabilityAgent(name="Fallback-Prob")
    
    # === DEEP LEARNING AGENTS ===
    elif difficulty == "dl-mlp":
        if os.path.exists("ml_models/battleship_mlp.h5") and os.path.exists("ml_models/battleship_scaler.pkl"):
            agent = MLPAgent(
                model_path="ml_models/battleship_mlp.h5",
                scaler_path="ml_models/battleship_scaler.pkl",
                features_path="ml_models/featureslist.txt",
                name="MLP-DeepLearning"
            )
            agent.agent_type = "dl"
            return agent
        return SmartProbabilityAgent(name="Fallback-Prob")

    # === REINFORCEMENT LEARNING AGENTS ===
    elif difficulty == "rl-qlearning":
        if os.path.exists("ml_models/q_table_features.pkl"):
            q_agent = QLearningAgent()
            with open("ml_models/q_table_features.pkl", "rb") as f:
                q_agent.q_table = pickle.load(f)
            q_agent.epsilon = 0.0
            wrapper = RLAgentWrapper(q_agent, name="Q-Learning")
            wrapper.agent_type = "rl"
            return wrapper
        return SmartProbabilityAgent(name="Fallback-Prob")

    elif difficulty == "rl-sarsa":
        if os.path.exists("ml_models/sarsa_table_features.pkl"):
            sarsa_agent = SARSAAgent()
            with open("ml_models/sarsa_table_features.pkl", "rb") as f:
                sarsa_agent.q_table = pickle.load(f)
            sarsa_agent.epsilon = 0.0
            wrapper = RLAgentWrapper(sarsa_agent, name="SARSA")
            wrapper.agent_type = "rl"
            return wrapper
        return SmartProbabilityAgent(name="Fallback-Prob")

    # === LLM AGENTS ===
    elif difficulty == "llm-local":
        agent = SmartLLMAgentV2(
            name="Llama-Local", 
            model="ianleelamb_d12d/battleship-lora-adapter", 
            client=local_client
        )
        return agent
    
    elif difficulty == "llm-openrouter":
        if os.getenv("OPENROUTER_API_KEY"):
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            agent = SmartLLMAgentV2(
                name="Llama-4-Scout",
                model="meta-llama/llama-4-scout",
                provider="openrouter",
                client=client
            )
            return agent
        return SmartProbabilityAgent(name="Fallback-Prob")
    
    # Default Fallback
    return SmartProbabilityAgent(name="Smart-Prob")

# --- AVAILABLE AGENTS ---
def get_available_agents():
    """Returns list of available agents based on what model files exist."""
    agents = []
    
    # Heuristic (always available)
    agents.append({"id": "smart-prob", "name": "Smart Probability Algorithm", "type": "heuristic", "available": True})
    agents.append({"id": "rule-based", "name": "Rule Based Agent", "type": "heuristic", "available": True})
    agents.append({"id": "heuristic", "name": "Heuristic Agent", "type": "heuristic", "available": True})
    
    # ML Agents
    agents.append({
        "id": "ml-logistic", 
        "name": "Logistic Regression", 
        "type": "ml", 
        "available": os.path.exists("ml_models/lr.joblib")
    })
    agents.append({
        "id": "ml-hgb", 
        "name": "Histogram Gradient Boosting", 
        "type": "ml", 
        "available": os.path.exists("ml_models/hgb.joblib")
    })
    agents.append({
        "id": "ml-lgbm", 
        "name": "LightGBM", 
        "type": "ml", 
        "available": os.path.exists("ml_models/lgbm.txt")
    })
    
    # Deep Learning
    agents.append({
        "id": "dl-mlp", 
        "name": "MLP Neural Network", 
        "type": "dl", 
        "available": os.path.exists("ml_models/battleship_mlp.h5") and os.path.exists("ml_models/battleship_scaler.pkl")
    })
    
    # Reinforcement Learning
    agents.append({
        "id": "rl-qlearning", 
        "name": "Q-Learning", 
        "type": "rl", 
        "available": os.path.exists("ml_models/q_table_features.pkl")
    })
    agents.append({
        "id": "rl-sarsa", 
        "name": "SARSA", 
        "type": "rl", 
        "available": os.path.exists("ml_models/sarsa_table_features.pkl")
    })
    
    # LLM Agents
    agents.append({
        "id": "llm-local", 
        "name": "Llama Local (Fine-tuned)", 
        "type": "llm", 
        "available": True  # Assumes local server might be running
    })
    agents.append({
        "id": "llm-openrouter", 
        "name": "Llama-4-Scout (OpenRouter)", 
        "type": "llm", 
        "available": bool(os.getenv("OPENROUTER_API_KEY"))
    })
    
    return agents

@app.get("/agents")
async def list_agents():
    """Return list of available agents for the client to display."""
    return get_available_agents()

# --- STARTUP & CLEANUP ---
@app.on_event("startup")
async def startup_event():
    init_db() 
    asyncio.create_task(cleanup_stale_sessions())

EXPIRATION_SECONDS = 3600 
async def cleanup_stale_sessions():
    while True:
        await asyncio.sleep(600) 
        current_time = time.time()
        to_delete = [gid for gid, s in games.items() if current_time - s.get("last_active", 0) > EXPIRATION_SECONDS]
        for gid in to_delete:
            del games[gid]
            print(f"🗑️ Cleaned up stale game: {gid}")

# --- GAME ENDPOINTS ---

@app.post("/start_game")
async def start_game(request: GameStartRequest):
    game_id = str(uuid.uuid4())
    
    agent = get_agent_by_difficulty(request.difficulty)
    
    if hasattr(agent, "reset_state"):
        agent.reset_state()
    
    # --- MODE LOGIC ---
    if request.mode == "race":
        # RACE MODE: Shared Puzzle Layout
        puzzle_board, puzzle_ships = place_ships(empty_board(), [5, 4, 3, 3, 2])
        
        # AI tries to solve this
        ai_target_board = copy.deepcopy(puzzle_board)
        
        # Player tries to solve this (Exact same layout)
        player_target_board = copy.deepcopy(puzzle_board) 
        
        # Create separate ship tracking for AI and Player (same positions, separate hit tracking)
        ai_ship_objects = [Ship(s.size, list(s.cells)) for s in puzzle_ships]
        player_ship_objects = [Ship(s.size, list(s.cells)) for s in puzzle_ships]
        
        session_data = {
            "mode": "race",
            "agent": agent,
            
            # AI's Game
            "ai_board": ai_target_board,          
            "ai_ships": ai_ship_objects,             
            "ai_ships_remaining": [5, 4, 3, 3, 2],
            "ai_view": [['.' for _ in range(10)] for _ in range(10)], 
            "ai_hits": 0, 
            
            # Player's Game
            "player_board": player_target_board,
            "player_ships": player_ship_objects,  
            "player_ships_remaining": [5, 4, 3, 3, 2], 
            "player_hits": 0, 
            
            "game_over": False,
            "winner": None,
            "logs": [],
            "turns": 0,
            "last_active": time.time()
        }

    else:
        # CLASSIC MODE
        ai_target_board, ai_target_ships = place_ships(empty_board(), [5, 4, 3, 3, 2])
        
        if not request.player_ships:
            # Fallback for dev/testing - random placement
            player_board, player_ship_objects = place_ships(empty_board(), [5, 4, 3, 3, 2])
        else:
            # Player provided ship placements - create board and Ship objects
            player_board = empty_board()
            player_ship_objects = []
            for ship_data in request.player_ships:
                cells = []
                for r, c in ship_data.coords:
                    if 0 <= r < 10 and 0 <= c < 10:
                        player_board[r][c] = "_"
                        cells.append((r, c))
                # Create Ship object for tracking hits
                player_ship_objects.append(Ship(ship_data.size, cells))
        
        session_data = {
            "mode": "classic",
            "agent": agent,
            "ai_board": ai_target_board,          
            "ai_ships": ai_target_ships,
            "ai_ships_remaining": [5, 4, 3, 3, 2],
            "player_board": player_board,
            "player_ships": player_ship_objects,  # Track player's ships for sink detection
            "player_ships_remaining": [5, 4, 3, 3, 2],
            "agent_board_view": [['.' for _ in range(10)] for _ in range(10)], 
            "game_over": False,
            "winner": None,
            "logs": [],
            "turns": 0,
            "last_active": time.time()
        }

    games[game_id] = session_data
    return {"game_id": game_id, "mode": request.mode, "message": "Game On!"}

@app.get("/game_state/{game_id}")
async def get_game_state(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game session expired or not found")
    
    session = games[game_id]
    session["last_active"] = time.time()
    
    if session["mode"] == "race":
        target_board = session["player_board"]
    else:
        target_board = session["ai_board"]

    masked_board = []
    for row in target_board:
        masked_row = [cell if cell in ["X", "O"] else "." for cell in row]
        masked_board.append(masked_row)

    response = {
        "game_id": game_id,
        "mode": session["mode"],
        "masked_board": masked_board,
        "logs": session["logs"],
        "game_over": session["game_over"],
        "agent_name": session["agent"].name,
        # Expose agent reasoning for frontend display when available
        "agent_last_reasoning": getattr(session.get("agent"), "last_reasoning", None),
        "agent_last_prompt": getattr(session.get("agent"), "last_prompt", None)
    }
    
    if session["mode"] == "race":
        response["race_stats"] = {
            "player_hits": session.get("player_hits", 0),
            "ai_hits": session.get("ai_hits", 0),
            "player_ships_left": len(session.get("player_ships_remaining", [])),
            "ai_ships_left": len(session.get("ai_ships_remaining", []))
        }
    else:
        response["ai_ships_remaining"] = session["ai_ships_remaining"]

    return response

@app.post("/play_turn")
async def play_turn(request: PlayerTurnRequest):
    game_id = request.game_id
    if game_id not in games: raise HTTPException(404, "No game")
    session = games[game_id]
    agent = session["agent"]
    session["last_active"] = time.time()
    
    if session["game_over"]:
        return {"game_over": True, "winner": session["winner"]}

    # --- 1. PROCESS PLAYER SHOT ---
    target_board_key = "player_board" if session["mode"] == "race" else "ai_board"
    
    try:
        col = request.player_shot[0].upper(); row = int(request.player_shot[1:])
        c = "ABCDEFGHIJ".index(col); r = row
    except: raise HTTPException(400, "Bad coords")

    p_result = "MISS"
    p_sunk_ship = None
    target_board = session[target_board_key]
    
    if not (0 <= r < 10 and 0 <= c < 10):
         raise HTTPException(400, "Coordinates out of bounds")

    if target_board[r][c] in [".", "_"]:
        if target_board[r][c] == "_":
            target_board[r][c] = "X"
            p_result = "HIT"
            
            # Determine which ships to check based on mode
            # Race: player shoots at player_board (puzzle) -> player_ships
            # Classic: player shoots at ai_board -> ai_ships
            ships_key = "player_ships" if session["mode"] == "race" else "ai_ships"
            remaining_key = "player_ships_remaining" if session["mode"] == "race" else "ai_ships_remaining"
            
            if session["mode"] == "race": 
                session["player_hits"] += 1
            
            # Register hit and check for sink
            for ship in session.get(ships_key, []):
                if ship.register_hit(r, c) and ship.is_sunk():
                    p_sunk_ship = ship.size
                    # Remove from remaining
                    if ship.size in session[remaining_key]:
                        session[remaining_key].remove(ship.size)
                    break
        elif target_board[r][c] == ".":
            target_board[r][c] = "O"
        
        session["turns"] += 1

    # --- 2. AI TURN ---
    ai_target_board_key = "ai_board" if session["mode"] == "race" else "player_board"
    ai_target_board = session[ai_target_board_key]
    
    # Update AI Memory (Simulated)
    if getattr(agent, "last_move", None):
        lr, lc = agent.last_move
        real_cell = ai_target_board[lr][lc]
        did_hit = real_cell in ["_", "X"]
        
        res_obj = {'hit': did_hit, 'sunk_ship': None} # Simplified sunk handling
        
        if hasattr(agent, 'update_state'):
            try:
                agent.update_state(agent.last_move, res_obj, getattr(agent,'agent_board',[]))
            except TypeError: pass
            
        view_key = "ai_view" if session["mode"] == "race" else "agent_board_view"
        if view_key in session:
            session[view_key][lr][lc] = 'h' if did_hit else 'm'

    # Generate Move
    ai_remaining_key = "ai_ships_remaining" if session["mode"] == "race" else "player_ships_remaining"
    
    if hasattr(agent, 'select_move_async'):
        # Protect against LLMs or async agents that may hang or error out.
        try:
            # Use a conservative timeout for external LLM calls
            ai_move_coords = await asyncio.wait_for(
                agent.select_move_async(getattr(agent, 'agent_board', []), session[ai_remaining_key]),
                timeout=8.0
            )
            agent.last_move = ai_move_coords
        except asyncio.TimeoutError:
            # Fallback: use agent-provided fallback if available, else pick first unknown
            try:
                if hasattr(agent, 'get_fallback_move'):
                    coverage = [[0]*10 for _ in range(10)]
                    ai_move_coords = agent.get_fallback_move(coverage, getattr(agent, 'agent_board', []))
                else:
                    # Pick first unknown cell
                    board = getattr(agent, 'agent_board', [['.']*10 for _ in range(10)])
                    found = False
                    for rr in range(10):
                        for cc in range(10):
                            if board[rr][cc] == '.':
                                ai_move_coords = (rr, cc)
                                found = True
                                break
                        if found: break
            except Exception as e:
                pass
                ai_move_coords = (0, 0)
            agent.last_move = ai_move_coords
        except Exception:
            # Best-effort fallback
            try:
                if hasattr(agent, 'get_fallback_move'):
                    coverage = [[0]*10 for _ in range(10)]
                    ai_move_coords = agent.get_fallback_move(coverage, getattr(agent, 'agent_board', []))
                else:
                    ai_move_coords = (0, 0)
            except Exception:
                ai_move_coords = (0, 0)
            agent.last_move = ai_move_coords
    else:
        board_view = session["ai_view"] if session["mode"] == "race" else session["agent_board_view"]
        ai_move_coords = agent.select_move(board_view, features={}, remaining=session[ai_remaining_key], turn=session["turns"])
        agent.last_move = ai_move_coords

    # Execute AI Move
    ar, ac = ai_move_coords
    ai_shot_result = "MISS"
    ai_sunk_ship = None
    
    if 0 <= ar < 10 and 0 <= ac < 10:
        if ai_target_board[ar][ac] == "_":
            ai_target_board[ar][ac] = "X"
            ai_shot_result = "HIT"
            
            # Determine which ships to check based on mode
            # Race: AI shoots at ai_board (puzzle) -> ai_ships
            # Classic: AI shoots at player_board -> player_ships
            ai_ships_key = "ai_ships" if session["mode"] == "race" else "player_ships"
            ai_remaining_key_for_sink = "ai_ships_remaining" if session["mode"] == "race" else "player_ships_remaining"
            
            if session["mode"] == "race": 
                session["ai_hits"] += 1
            
            # Register hit and check for sink
            for ship in session.get(ai_ships_key, []):
                if ship.register_hit(ar, ac) and ship.is_sunk():
                    ai_sunk_ship = ship.size
                    # Remove from remaining
                    if ship.size in session[ai_remaining_key_for_sink]:
                        session[ai_remaining_key_for_sink].remove(ship.size)
                    break
        elif ai_target_board[ar][ac] == ".":
            ai_target_board[ar][ac] = "O"
            ai_shot_result = "MISS"
        
    ai_move_str = f"{'ABCDEFGHIJ'[ac]}{ar}"

    # Response with RACE STATS
    response_data = {
        "player_shot_result": p_result,
        "player_sunk_ship": p_sunk_ship,  # Size of ship player sunk (if any)
        "ai_shot_result": ai_shot_result,
        "ai_sunk_ship": ai_sunk_ship,  # Size of ship AI sunk (if any)
        "ai_move": ai_move_str,
        "game_over": False,
        # Agent reasoning/prompt to help client show explanation
        "agent_last_reasoning": getattr(session.get("agent"), "last_reasoning", None),
        "agent_last_prompt": getattr(session.get("agent"), "last_prompt", None)
    }

    # (No console logging here) Keep responses tidy for production.
    
    if session["mode"] == "race":
        response_data["race_stats"] = {
            "player_hits": session.get("player_hits", 0),
            "ai_hits": session.get("ai_hits", 0),
            "player_ships_left": len(session.get("player_ships_remaining", [])),
            "ai_ships_left": len(session.get("ai_ships_remaining", [])),
            "player_ships_remaining": session.get("player_ships_remaining", []),
            "ai_ships_remaining": session.get("ai_ships_remaining", [])
        }
        
        # Check Win Conditions for Race
        if session["ai_hits"] >= 17:
            session["game_over"] = True
            session["winner"] = "AI"
            add_score(session["agent"].name, session["turns"], session["agent"].name, "race", "agent")
            return {**response_data, "game_over": True, "winner": "AI"}
            
        if session["player_hits"] >= 17:
            session["game_over"] = True
            session["winner"] = "Player"
            return {**response_data, "game_over": True, "winner": "Player"}
            
    else:
        # Classic Mode Stats
        response_data["classic_stats"] = {
            "ai_ships_remaining": session.get("ai_ships_remaining", []),
            "player_ships_remaining": session.get("player_ships_remaining", [])
        }
        
        # Classic Win Logic - check if all ships are sunk
        if len(session.get("ai_ships_remaining", [])) == 0:
            session["game_over"] = True
            session["winner"] = "Player"
            return {**response_data, "game_over": True, "winner": "Player"}
            
        if len(session.get("player_ships_remaining", [])) == 0:
            session["game_over"] = True
            session["winner"] = "AI"
            add_score(session["agent"].name, session["turns"], session["agent"].name, "classic", "agent")
            return {**response_data, "game_over": True, "winner": "AI"}

    return response_data

# --- LEADERBOARD ENDPOINTS ---

@app.get("/leaderboard")
async def get_leaderboard_data(mode: str = "classic", type: str = "human"):
    return get_top_scores(limit=10, mode=mode, player_type=type)

@app.post("/submit_score")
async def submit_score_endpoint(request: ScoreSubmission):
    game_id = request.game_id
    if game_id not in games: raise HTTPException(404, "No game")
    session = games[game_id]
    
    if not session["game_over"]: raise HTTPException(400, "Game not over")
        
    add_score(request.player_name, session["turns"], session["agent"].name, session["mode"], "human")
    return {"message": "Score saved"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)