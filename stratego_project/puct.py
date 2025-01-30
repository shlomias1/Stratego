import math
from stratego import Stratego
from game_net import GameNetwork

class PUCTNode:
    def __init__(self, state, parent=None, move=None, prior_prob=1.0):
        """
        Node in the PUCT search tree.
        :param state: The game state.
        :param parent: The parent node.
        :param move: The move that led to this state.
        :param prior_prob: The prior probability from the policy network.
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}  # Dictionary of move -> child node
        self.P = prior_prob  # Prior probability from policy network
        self.Q = 0  # Average value of this node
        self.N = 0  # Visit count

    def select_child(self, cpuct):
        """
        Select a child node using the PUCT formula.
        """
        best_score = -float("inf")
        best_child = None
        sqrt_N_parent = math.sqrt(self.N + 1)  # Avoid division by zero

        for move, child in self.children.items():
            # Compute PUCT score
            U = child.Q + cpuct * child.P * (sqrt_N_parent / (1 + child.N))
            if U > best_score:
                best_score = U
                best_child = child

        return best_child

    def expand(self, game, policy_probs):
        """
        Expand the node by adding child nodes for legal moves.
        :param game: The current game state.
        :param policy_probs: Dictionary of move -> probability from policy network.
        """
        legal_moves = game.legal_moves()
        for move in legal_moves:
            if move not in self.children:
                prior_prob = policy_probs.get(move, 1e-4)  # Small probability for unseen moves
                new_state = game.clone()
                new_state.make_move(*move)
                self.children[move] = PUCTNode(new_state, parent=self, move=move, prior_prob=prior_prob)

    def update(self, value):
        """
        Update Q-value using incremental averaging.
        :param value: Value estimate (-1 to 1).
        """
        self.N += 1
        self.Q += (value - self.Q) / self.N  # Running average of Q

    def is_fully_expanded(self):
        """Check if all legal moves have been expanded."""
        return len(self.children) == len(self.state.legal_moves())

    def best_child(self):
        """Return the child with the highest visit count (N)."""
        return max(self.children.values(), key=lambda c: c.N, default=None)

class PUCTPlayer:
    def __init__(self, network, simulations=800, cpuct=1.5):
        """
        PUCT Player using a neural network.
        :param network: The neural network for policy and value estimation.
        :param simulations: Number of simulations per move.
        :param cpuct: Exploration constant in PUCT formula.
        """
        self.network = network
        self.simulations = simulations
        self.cpuct = cpuct

    def choose_move(self, game):
        """
        Perform PUCT search and choose the best move.
        """
        root = PUCTNode(game.clone())
        state_tensor = torch.tensor(game.encode(), dtype=torch.float32).unsqueeze(0)
        policy_probs, value_estimate = self.network(state_tensor)
        legal_moves = game.legal_moves()
        policy_dict = {move: policy_probs[0][idx].item() for idx, move in enumerate(legal_moves)}
        root.expand(game, policy_dict)
        root.update(value_estimate.item())  
        for _ in range(self.simulations):
            node = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)
        best_child = root.best_child()
        return best_child.move if best_child else None

    def _select(self, node):
        """Traverse the tree using PUCT selection until reaching a leaf."""
        while node.is_fully_expanded():
            node = node.select_child(self.cpuct)
        return node

    def _evaluate(self, node):
        """
        Evaluate a leaf node using the neural network.
        :return: Value estimate (-1 to 1).
        """
        state_tensor = torch.tensor(node.state.encode(), dtype=torch.float32).unsqueeze(0)
        _, value_estimate = self.network(state_tensor)
        return value_estimate.item()

    def _backpropagate(self, node, value):
        """
        Backpropagate the value estimate up the tree.
        """
        while node:
            node.update(value)
            node = node.parent