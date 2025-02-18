import random
import os
import gc
import json
import torch
import torch.optim as optim
import numpy as np
from stratego import Stratego
from mcts import MCTSPlayer
from game_net import GameNetwork
from utils import _create_log, Connect_CUDA
import config

class PreTrain:
    def __init__(self, num_games=10000, filename="games_data.json"):
        self.num_games = num_games
        self.filename = filename 

    def generate_self_play_games(self):
        games = []
        batch_size = 5
        for i in range(self.num_games):
            game = Stratego()
            log_message = f"🔄 Generating game {i+1}/{self.num_games}..."
            _create_log(log_message, "Info","game_generation_log.txt")
            print(log_message) 
            mcts_player = MCTSPlayer(simulations = config.MCTS_SIMULATIONS, exploration_weight=config.EXPLORATION_WEIGHT)
            game.auto_place_pieces_for_player("red")  
            game.auto_place_pieces_for_player("blue")
            game_history = []
            move_count = 0
            MAX_MOVES = config.MAX_MOVES
            while not game.game_over:
                move = mcts_player.choose_move(game)
                if move is None:
                    print("⚠️ No valid move found. Skipping turn.")
                    break  
                game.make_move(*move)
                game_history.append(game.encode())  
                move_count += 1
                if move_count >= MAX_MOVES:
                    print("🏁 Draw - you have reached the move limit!")
                    game.game_over = True
            games.append((game_history, game.status())) 
            if (i + 1) % batch_size == 0:
                self.save_games_data(games)
                games.clear() 
                gc.collect()              
        if games:
            self.save_games_data(games)
        print("✅ All self-play games have been generated and saved.")

    def prepare_training_data(self, games_data):
        total_games = len(games_data)
        feature_size = len(games_data[0][0])
        policy_size = len(game_history[0])
        inputs = np.memmap("inputs.dat", dtype="float16", mode="w+", shape=(total_games, feature_size))
        policy_labels = np.memmap("policy_labels.dat", dtype="float16", mode="w+", shape=(total_games, policy_size))
        value_labels = np.memmap("value_labels.dat", dtype="float16", mode="w+", shape=(total_games,))
        for game_history, outcome in games_data:
            for state_vector in game_history:
                inputs.append(state_vector)
                if outcome == "red wins":
                    value_labels.append(1)
                elif outcome == "blue wins":
                    value_labels.append(-1)
                else:
                    value_labels.append(0)
                policy_probs = [random.random() for _ in range(len(game_history[0]))] # Mock for now
                total_visits = sum(policy_probs)
                policy_probs = [p / total_visits for p in policy_probs]
                policy_labels.append(policy_probs)
        return np.array(inputs), np.array(policy_labels), np.array(value_labels)

    def save_games_data(self, new_games):        
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
        filename = self.filename
        if not new_games:
            log_msg = "⚠️ No new games to save. Skipping write operation."
            _create_log(log_msg, "Warning", "game_generation_log.txt")
            print(log_msg)
            return
        with open(filename, "a") as f:
            for game in new_games:
                f.write(json.dumps(convert_to_serializable(game)) + "\n")    
        num_saved_games = len(new_games)
        new_games.clear()
        gc.collect() 
        log_msg = f"✅ {num_saved_games} new games saved in append mode. Memory cleared."
        _create_log(log_msg, "Info", "game_generation_log.txt")
        print(log_msg)

    def load_games_data(self):
        filename = self.filename
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            log_message = "❌ No saved games found or file is empty, generating new games..."
            _create_log(log_message, "Warning")
            print(log_message)
            return None
        games_data = []
        try:
            with open(filename, "r") as f:
                for line_number, line in enumerate(f, start=1):
                    try:
                        games_data.append(json.loads(line.strip())) 
                    except json.JSONDecodeError as e:
                        log_msg = f"⚠️ Skipping corrupted line {line_number}: {e}"
                        _create_log(log_msg, "Warning")
                        print(log_msg)
            if not games_data:
                log_msg = "❌ Saved file is empty or all lines were corrupted."
                _create_log(log_msg, "Warning")
                print(log_msg)
                return None
            print(f"✅ Loaded {len(games_data)} games from {filename}")
            return games_data
        except Exception as e:
            log_msg = f"❌ Error opening JSON file: {e}"
            _create_log(log_msg, "Error")
            print(log_msg)
            return None
        
class Train:
    def __init__(self, network, inputs, policy_labels, value_labels, epochs=10, batch_size=16):
        self.network = network
        self.inputs = inputs
        self.policy_labels = policy_labels
        self.value_labels = value_labels
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        device = Connect_CUDA()
        optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)
        for epoch in range(self.epochs):
            total_correct = 0
            total_samples = 0
            for i in range(0, len(self.inputs), self.batch_size):
                batch_inputs = torch.tensor(self.inputs[i:i+self.batch_size], dtype=torch.float16, device=device)
                batch_policy_labels = torch.tensor(self.policy_labels[i:i+self.batch_size], dtype=torch.float16)
                batch_value_labels = torch.tensor(self.value_labels[i:i+self.batch_size], dtype=torch.float16)
                optimizer.zero_grad()
                with torch.no_grad():
                    policy_probs, value_estimate = self.network(batch_inputs)
                if policy_probs.shape != batch_policy_labels.shape:
                    log_msg = "Shape mismatch between policy_probs and batch_policy_labels!"
                    _create_log(log_msg, "Error")
                    print(log_msg)
                    return
                policy_loss = -torch.sum(batch_policy_labels * torch.log(policy_probs)) / self.batch_size
                value_loss = torch.mean((batch_value_labels - value_estimate) ** 2)
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
                predicted_classes = torch.argmax(policy_probs, dim=1)
                true_classes = torch.argmax(batch_policy_labels, dim=1)
                correct = (predicted_classes == true_classes).sum().item()
                total_correct += correct
                total_samples += batch_policy_labels.shape[0]
            accuracy = (total_correct / total_samples) * 100
            log_line1 = f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
            log_line2 = f"🔍 Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            log_line3 = f"🔍 Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            _create_log(log_line1, "Info", "training_log.txt")
            _create_log(log_line2, "Info", "training_log.txt")
            _create_log(log_line3, "Info", "training_log.txt")
            print(log_line1)
            print(log_line2)
            print(log_line3)
            torch.cuda.empty_cache()
        self.network.save_model("trained_game_network.pth")
