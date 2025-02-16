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

class PreTrain:
    def __init__(self, num_games=10000, filename="games_data.json"):
        self.num_games = num_games
        self.filename = filename 

    def generate_self_play_games(self):
        games = []
        batch_size = 20
        for i in range(self.num_games):
            game = Stratego()
            print(f"🔄 Generating game {i+1}/{self.num_games}...")
            mcts_player = MCTSPlayer(simulations=700, exploration_weight=2)
            game.auto_place_pieces_for_player("red")  
            game.auto_place_pieces_for_player("blue")
            game_history = []
            move_count = 0
            MAX_MOVES = 700
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
        inputs = []
        policy_labels = []
        value_labels = []
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
        all_games = []
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as f:
                try:
                    all_games = json.load(f)
                except json.JSONDecodeError:
                    print("⚠️ Warning: JSON file is corrupt. Creating a new file.")
        all_games.extend(convert_to_serializable(new_games))
        with open(filename, "w") as f:
            json.dump(all_games, f)
        print(f"✅ {len(new_games)} new games saved. Total games in file: {len(all_games)}")

    def load_games_data(self):
        try:
            if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
                print("❌ No saved games found or file is empty, generating new games...")
                return None
            with open(self.filename, "r") as f:
                games_data = json.load(f)
            if not games_data: 
                print("❌ Saved file is empty, generating new games...")
                return None
            print(f"✅ Loaded {len(games_data)} games from {self.filename}")
            return games_data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error loading JSON file: {e}")
            return None
        
class Train:
    def __init__(self, network, inputs, policy_labels, value_labels, epochs=10, batch_size=32):
        self.network = network
        self.inputs = inputs
        self.policy_labels = policy_labels
        self.value_labels = value_labels
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        for epoch in range(self.epochs):
            total_correct = 0
            total_samples = 0
            for i in range(0, len(self.inputs), self.batch_size):
                batch_inputs = torch.tensor(self.inputs[i:i+self.batch_size], dtype=torch.float32)
                batch_policy_labels = torch.tensor(self.policy_labels[i:i+self.batch_size], dtype=torch.float32)
                batch_value_labels = torch.tensor(self.value_labels[i:i+self.batch_size], dtype=torch.float32)
                optimizer.zero_grad()
                policy_probs, value_estimate = self.network(batch_inputs)
                if policy_probs.shape != batch_policy_labels.shape:
                    print("❌ Error: Shape mismatch between policy_probs and batch_policy_labels!")
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
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        self.network.save_model("trained_game_network.pth")
