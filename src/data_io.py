import json
import numpy as np
import gc
import os
import pandas as pd
from utils import _create_log
import config

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj] 
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj) 
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()} 
    else:
        return obj

def save_data_to_JSON(new_games,filename):        
    if not new_games:
        log_msg = "⚠️ No new games to save. Skipping write operation."
        _create_log(log_msg, "Warning", "game_generation_log.txt")
        print(log_msg)
        return
    with open(filename, "a") as f:
        for game_history, status in new_games:
            game_dict = {
                "encoded_state": convert_to_serializable(game_history),
                "status": status
            }
            f.write(json.dumps(game_dict) + "\n")   
    num_saved_games = len(new_games)
    new_games.clear()
    gc.collect() 
    log_msg = f"✅ {num_saved_games} new games saved in append mode. Memory cleared."
    _create_log(log_msg, "Info", "game_generation_log.txt")
    print(log_msg)

def save_data_to_mem(new_games, memmap_file, filename):
    if not new_games:
        log_msg = "⚠️ No new games to save. Skipping write operation."
        _create_log(log_msg, "Warning", "game_generation_log.txt")
        print(log_msg)
        return
    feature_size = config.FEATURE_VECTOR_SIZE
    existing_games = 0
    if os.path.exists(memmap_file):
        existing_memmap = np.memmap(memmap_file, dtype="float16", mode="r")
        existing_games = existing_memmap.shape[0] // feature_size  
    total_games = existing_games + len(new_games)
    inputs_memmap = np.memmap(memmap_file, dtype="float16", mode="w+", shape=(total_games, feature_size))
    with open(filename, "a") as f:
        for i, (game_history, status) in enumerate(new_games):
            if isinstance(game_history, np.ndarray):
                game_history = game_history.tolist()
            f.write(json.dumps({"encoded_state": convert_to_serializable(game_history), "status": status}) + "\n")  
            inputs_memmap[existing_games + i, :] = np.array(game_history, dtype="float16") 
    inputs_memmap.flush() 
    log_msg = f"✅ {len(new_games)} new games saved. Total games stored: {total_games}."
    _create_log(log_msg, "Info", "game_generation_log.txt")
    print(log_msg)

def load_data_from_JSON(filename, memmap_file="games_memmap.dat"):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        log_message = "❌ No saved games found or file is empty, generating new games..."
        _create_log(log_message, "Warning")
        return None
    total_states = 0
    total_games = 0
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    total_states += len(json.loads(line.strip())["encoded_state"])
                    total_games +=1
                except json.JSONDecodeError as e:
                    _create_log(f"⚠️ Skipping corrupted line: {e}", "Error")
                    continue
    except Exception as e:
        _create_log(f"⚠️ Error opening JSON file: {e}", "Error")
        return None
    feature_size = config.FEATURE_VECTOR_SIZE
    inputs_memmap = np.memmap(memmap_file, dtype="float16", mode="w+", shape=(total_states, feature_size))
    state_index = 0
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    game_entry = json.loads(line.strip()) 
                    for state_vector in game_entry["encoded_state"]:
                        inputs_memmap[state_index] = state_vector  
                        state_index += 1  
                except json.JSONDecodeError as e:
                    _create_log(f"⚠️ Skipping corrupted line: {e}", "Error")
                    continue
    except Exception as e:
        _create_log(f"⚠️ Error reading JSON file: {e}", "Error")
        return None
    inputs_memmap.flush()
    _create_log(f"✅ Loaded {total_games} games from {filename} into {memmap_file}.", "Info")
    return inputs_memmap 

# def load_data_from_JSON(filename):
#     if not os.path.exists(filename) or os.path.getsize(filename) == 0:
#         log_message = "❌ No saved games found or file is empty, generating new games..."
#         _create_log(log_message, "Warning")
#         return None        
#     games_data = []
#     try:
#         with open(filename, "r") as f:
#             for line in f: 
#                 try:
#                     game_data = json.loads(line.strip())
#                     games_data.append(game_data)
#                 except json.JSONDecodeError as e:
#                     log_msg = f"⚠️ Skipping corrupted line: {e}"
#                     _create_log(log_msg, "Error")
#                     continue
#         _create_log(f"✅ Loaded {len(games_data)} games from {filename}", "Info")
#         return games_data  
#     except Exception as e:
#         log_msg = f"⚠️ Error opening JSON file: {e}"
#         _create_log(log_msg, "Error")
#         return None

def load_data_from_mem(memmap_file):
    if not os.path.exists(memmap_file):
        log_message = "❌ No saved games found in Memmap."
        _create_log(log_message, "Warning")
        print(log_message)
        return None
    inputs_memmap = np.memmap(memmap_file, dtype="float16", mode="r")
    feature_size = config.FEATURE_VECTOR_SIZE  
    total_games = inputs_memmap.shape[0] // feature_size  
    log_msg = f"✅ Loaded {total_games} games from {memmap_file}."
    _create_log(log_msg, "Info")
    print(log_msg)
    for i in range(total_games):
        yield inputs_memmap[i] 