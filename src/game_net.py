import torch
import torch.nn as nn
import torch.optim as optim
from config import HIDDEN_LAYER_SIZE

class GameNetwork(nn.Module):
    def __init__(self, input_dim, policy_output_dim, value_output_dim = 1, hidden_dim = HIDDEN_LAYER_SIZE):
        """
        Neural network for predicting policy (best move) and value (win probability)
        :param input_dim: Size of input feature vector (output of `encode`)
        :param policy_output_dim: Number of possible actions (output of `decode`)
        :param value_output_dim: Output size of the value head (1 for win probability)
        :param hidden_dim: Number of neurons in hidden layers
        """
        super(GameNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head (predicts move probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, policy_output_dim),
            nn.Softmax(dim=-1)  # Ensures output is a probability distribution
        )

        # Value head (predicts win probability)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, value_output_dim),
            nn.Tanh()  # Outputs value between -1 (losing) and 1 (winning)
        )

    def forward(self, x):
        """
        Forward pass of the network.
        :param x: Input tensor of game state
        :return: policy_probs (action probabilities), value_estimate (win probability)
        """
        features = self.feature_extractor(x)
        policy_probs = self.policy_head(features)
        value_estimate = self.value_head(features)
        return policy_probs, value_estimate

    def save_model(self, file_path="game_network.pth"):
        """
        Save model weights to a file.
        :param file_path: File path to save model
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path="game_network.pth"):
        """
        Load model weights from a file.
        :param file_path: File path to load model from
        """
        self.load_state_dict(torch.load(file_path))
        self.eval()  # Set model to evaluation mode
        print(f"Model loaded from {file_path}")
