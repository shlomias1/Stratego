import random
import os
import gc
import json
import torch
import torch.optim as optim
import numpy as np
from stratego import Stratego
from mcts import MCTSPlayer
from utils import _create_log, Connect_CUDA
import config
import pandas as pd
import data_io

class PreTrain:
    def __init__(self, num_games=10000, filename="games_data.json", memmap_file="games_memmap.dat"):
        self.num_games = num_games
        self.filename = filename 
        self.memmap_file = memmap_file

    def generate_self_play_games(self):
        games = []
        batch_size = 5
        for i in range(self.num_games):
            game = Stratego()
            red_player = MCTSPlayer(simulations=config.MCTS_SIMULATIONS, exploration_weight=config.EXPLORATION_WEIGHT)
            blue_player = MCTSPlayer(simulations=config.MCTS_SIMULATIONS, exploration_weight=config.EXPLORATION_WEIGHT)
            game.auto_place_pieces_for_player("red")  
            game.auto_place_pieces_for_player("blue")
            game_history = []
            move_count = 0
            MAX_MOVES = config.MAX_MOVES
            while not game.game_over:
                current_player = red_player if game.turn == "red" else blue_player
                move = current_player.choose_move(game)
                if move is None:
                    print("⚠️ No valid move found. Skipping turn.")
                    break  
                game.make_move(*move)
                game_history.append(game.encode())  
                move_count += 1
                if move_count >= MAX_MOVES:
                    print("🏁 Draw - you have reached the move limit!")
                    game.game_over = True
            log_message = f"🔄 Generating game {i+1}/{self.num_games} - status: {game.status()} after {MAX_MOVES} moves"
            _create_log(log_message, "Info","game_generation_log.txt")
            if(game.status() == "red wins"): 
                status = 1
            elif(game.status() == "blue wins"): 
                status = -1
            else:
                status = 0         
            games.append((game_history, status)) 
            if (i + 1) % batch_size == 0:
                data_io.save_data_to_JSON(games,self.filename)
                games.clear() 
                gc.collect()              
        if games:
            data_io.save_data_to_JSON(games,self.filename)
        print("✅ All self-play games have been generated and saved.")
    
    def load_games_data(self):
        if os.path.exists(self.filename):
            return data_io.load_data_from_JSON(self.filename)

    def prepare_training_data(self):
        if not os.path.exists(self.memmap_file):
            _create_log(f"❌ Memmap file {self.memmap_file} not found!", "Error")
            return None, None, None
        feature_size = config.FEATURE_VECTOR_SIZE
        total_states = os.path.getsize(self.memmap_file) // (feature_size * np.dtype("float16").itemsize)
        inputs = np.memmap(self.memmap_file, dtype="float16", mode="r", shape=(total_states, feature_size))
        policy_labels = np.memmap("policy_labels.dat", dtype="float16", mode="w+", shape=(total_states, feature_size))
        value_labels = np.memmap("value_labels.dat", dtype="float16", mode="w+", shape=(total_states,))
        for state in range(total_states):
            value_labels[state] = np.random.choice([-1, 0, 1])
            policy_probs = np.random.rand(feature_size)
            policy_probs /= policy_probs.sum()
            policy_labels[state] = policy_probs
        policy_labels.flush()
        value_labels.flush()
        _create_log(f"✅ Prepared training data from {total_states} states in {self.memmap_file}.", "Info")
        return inputs, policy_labels, value_labels
  
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
                batch_inputs = torch.tensor(self.inputs[i:i+self.batch_size], dtype=torch.float32, device=device, requires_grad=True)
                batch_policy_labels = torch.tensor(self.policy_labels[i:i+self.batch_size], dtype=torch.float32, device=device)
                batch_value_labels = torch.tensor(self.value_labels[i:i+self.batch_size], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                policy_probs, value_estimate = self.network(batch_inputs)
                if policy_probs.shape != batch_policy_labels.shape:
                    log_msg = "Shape mismatch between policy_probs and batch_policy_labels!"
                    _create_log(log_msg, "Error")
                    print(log_msg)
                    return
                policy_probs = policy_probs / policy_probs.sum(dim=-1, keepdim=True)
                policy_loss = -torch.sum(batch_policy_labels * torch.nn.functional.log_softmax(policy_probs, dim=-1)) / self.batch_size
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
            log_line = f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
            _create_log(log_line, "Info", "training_log.txt")
            print(log_line)
            torch.cuda.empty_cache()
        self.network.save_model("trained_game_network.pth")
