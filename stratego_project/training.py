import random
import torch
import torch.optim as optim
import numpy as np
from stratego import Stratego
from mcts import MCTSPlayer
from game_net import GameNetwork

class PreTrain:
    def __init__(self, num_games=10000):
        self.num_games = num_games

    def generate_self_play_games(self):
        games = []
        for _ in range(self.num_games):
            game = Stratego()
            mcts_player = MCTSPlayer(simulations=500, exploration_weight=2)
            game.auto_place_pieces_for_player("red")  # Automatically place pieces
            game.auto_place_pieces_for_player("blue")  # Automatically place pieces
            game_history = []
            while not game.game_over:
                if game.turn == "red":  # MCTS player (red)
                    move = mcts_player.choose_move(game)
                    game.make_move(*move)
                else:  # MCTS player (blue)
                    move = mcts_player.choose_move(game)
                    game.make_move(*move)
                
                game_history.append(game.encode())  # Store the game state

            games.append((game_history, game.status()))  # Record game history and outcome

        return games

    def prepare_training_data(self, games_data):
        inputs = []
        policy_labels = []
        value_labels = []

        for game_history, outcome in games_data:
            for state_vector in game_history:
                inputs.append(state_vector)

                # Value labels
                if outcome == "red wins":
                    value_labels.append(1)
                elif outcome == "blue wins":
                    value_labels.append(-1)
                else:
                    value_labels.append(0)

                # Policy labels (visit counts)
                # Here, you can calculate the policy based on visit counts or mock them.
                policy_probs = [random.random() for _ in range(len(game.legal_moves()))]  # Mock for now
                total_visits = sum(policy_probs)
                policy_probs = [p / total_visits for p in policy_probs]
                policy_labels.append(policy_probs)

        return np.array(inputs), np.array(policy_labels), np.array(value_labels)

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
            for i in range(0, len(self.inputs), self.batch_size):
                batch_inputs = torch.tensor(self.inputs[i:i+self.batch_size], dtype=torch.float32)
                batch_policy_labels = torch.tensor(self.policy_labels[i:i+self.batch_size], dtype=torch.float32)
                batch_value_labels = torch.tensor(self.value_labels[i:i+self.batch_size], dtype=torch.float32)
                optimizer.zero_grad()
                policy_probs, value_estimate = self.network(batch_inputs)
                policy_loss = -torch.sum(batch_policy_labels * torch.log(policy_probs)) / self.batch_size
                value_loss = torch.mean((batch_value_labels - value_estimate) ** 2)
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")
        self.network.save_model("trained_game_network.pth")