import numpy as np
import copy
import re
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

class Stratego:
    def __init__(self):
        self.board = np.full((10, 10), "EMPTY", dtype=object)
        self.turn = 'red'
        self.history = []
        self.game_over = False
        self.water = {(4, 2), (4, 3), (4, 6), (4, 7), (5, 2), (5, 3), (5, 6), (5, 7)}
        self.soldiers = {
            "red": {
                "FLAG": {"Rank": 0, "Quantity": 1},
                "BOMB": {"Rank": 0, "Quantity": 6},
                "SPY": {"Rank": 1, "Quantity": 1},
                "SCOUT": {"Rank": 2, "Quantity": 8},
                "SAPPER": {"Rank": 3, "Quantity": 5},
                "Sergeant": {"Rank": 4, "Quantity": 4},
                "Lieutenant": {"Rank": 5, "Quantity": 4},
                "Captain": {"Rank": 6, "Quantity": 4},
                "Major": {"Rank": 7, "Quantity": 3},
                "Colonel": {"Rank": 8, "Quantity": 2},
                "General": {"Rank": 9, "Quantity": 1},
                "MARSHAL": {"Rank": 10, "Quantity": 1},
            },
            "blue": {
                "FLAG": {"Rank": 0, "Quantity": 1},
                "BOMB": {"Rank": 0, "Quantity": 6},
                "SPY": {"Rank": 1, "Quantity": 1},
                "SCOUT": {"Rank": 2, "Quantity": 8},
                "SAPPER": {"Rank": 3, "Quantity": 5},
                "Sergeant": {"Rank": 4, "Quantity": 4},
                "Lieutenant": {"Rank": 5, "Quantity": 4},
                "Captain": {"Rank": 6, "Quantity": 4},
                "Major": {"Rank": 7, "Quantity": 3},
                "Colonel": {"Rank": 8, "Quantity": 2},
                "General": {"Rank": 9, "Quantity": 1},
                "MARSHAL": {"Rank": 10, "Quantity": 1},
            },
        }

        self.pieces = {
            color: {name: {"quantity": 0, "owner": color} for name in self.soldiers[color]} for color in self.soldiers
        }
        self.prob_board_red = {
            (row, col): {piece: 0 for piece in self.soldiers["blue"].keys()} for row in range(10) for col in range(10)
        }

        self.prob_board_blue = {
            (row, col): {piece: 0 for piece in self.soldiers["red"].keys()} for row in range(10) for col in range(10)
        }
        self.initialize_probabilities()

    def initialize_probabilities(self):
        for row in range(10):
            for col in range(10):
                if row in range(0, 4):
                    total_quantity = sum(self.soldiers["blue"][p]["Quantity"] for p in self.soldiers["blue"].keys())
                    for piece in self.soldiers["blue"].keys():
                        self.prob_board_red[(row, col)][piece] = self.soldiers["blue"][piece]["Quantity"] / total_quantity
                elif row in range(6, 10):
                    total_quantity = sum(self.soldiers["red"][p]["Quantity"] for p in self.soldiers["red"].keys())
                    for piece in self.soldiers["red"].keys():
                        self.prob_board_blue[(row, col)][piece] = self.soldiers["red"][piece]["Quantity"] / total_quantity
    
    def normalize_probabilities(self):
        for board in [self.prob_board_red, self.prob_board_blue]:
            for position in board:
                total_prob = sum(board[position].values())
                if total_prob > 0:
                    for piece in board[position]:
                        board[position][piece] /= total_prob

    def get_prob_board(self):
        return self.prob_board_red if self.turn == "red" else self.prob_board_blue

    def print_probabilities(self):
        for color, board in zip(["red", "blue"], [self.prob_board_red, self.prob_board_blue]):
            print(f"\n{color.upper()} PROBABILITIES:")
            for position, probs in board.items():
                print(f"{position}: {probs}")
                
    def input_pieces(self, player):
        total_pieces = sum(data["Quantity"] for data in self.soldiers[player].values())
        placed_pieces = 0
        valid_rows = range(6, 10) if player == 'red' else range(0, 4)
        print(f"Player {player}: Please place your pieces.")
        print(f"You can place your pieces only in rows {valid_rows.start}-{valid_rows.stop - 1}.")
        while placed_pieces < total_pieces:
            self.display_remaining_pieces(player)
            piece_name = input("Enter the piece name (e.g., Flag, Bomb, Spy): ").strip().upper()
            if piece_name not in self.soldiers[player]:
                print(f"Invalid piece name: {piece_name}. Please try again.")
                continue
            try:
                row = int(input(f"Enter the row ({valid_rows.start}-{valid_rows.stop - 1}): "))
                if row not in valid_rows:
                    raise ValueError(f"Invalid row. {player.capitalize()} pieces must be placed in rows {valid_rows.start}-{valid_rows.stop - 1}.")
                col = int(input("Enter the column (0-9): "))
                self.place_piece(player, piece_name, row, col)
                placed_pieces += 1
                print(f"The piece {piece_name} was successfully placed at position ({row}, {col}).")
                print("\nBoard after placement:")
                print(self)  # Print the board after each placement
            except ValueError as e:
                print(e)
        print(f"\nPlayer {player} has finished placing all pieces.")

    def display_remaining_pieces(self, player):
        print(f"\nRemaining pieces for player {player}:")
        for name, data in self.soldiers[player].items():
            max_quantity = data["Quantity"]
            placed = self.pieces[player][name]["quantity"]
            remaining = max_quantity - placed
            print(f"{name}: {remaining} out of {max_quantity}")

    def place_piece(self, player, piece_name, row, col):
        if self.pieces[player][piece_name]["quantity"] >= self.soldiers[player][piece_name]["Quantity"]:
            raise ValueError(f"Cannot place more {piece_name}. Maximum reached.")
        if row not in (range(6, 10) if player == "red" else range(0, 4)):
            raise ValueError(f"{player.capitalize()} must place pieces in their designated rows.")
        if (row, col) in self.water:
            raise ValueError("Cannot place a piece on water tiles.")
        if self.board[row][col] != "EMPTY":
            raise ValueError("Position is already occupied.")
        self.board[row][col] = f"{piece_name}_{player}"
        self.pieces[player][piece_name]["quantity"] += 1

    def auto_place_pieces_for_player(self, player):
        if player not in ['red', 'blue']:
            print("Invalid player. Choose 'red' or 'blue'.")
            return
        rows = range(6, 10) if player == "red" else range(0, 4)
        flag_position = (rows.stop - 1, 0)
        self.board[flag_position] = f"FLAG_{player}"
        self.pieces[player]["FLAG"]["quantity"] += 1
        bomb_positions = [(flag_position[0] - 1, flag_position[1]),
                        (flag_position[0], flag_position[1] + 1),
                        (flag_position[0] - 1, flag_position[1] + 1)]
        for row, col in bomb_positions:
            if row in rows and col < 10:
                self.board[row, col] = f"BOMB_{player}"
                self.pieces[player]["BOMB"]["quantity"] += 1
        positions = [(row, col) for row in rows for col in range(10) if self.board[row, col] == "EMPTY"]
        random.shuffle(positions)
        piece_priority = ["MARSHAL", "General", "Colonel", "Major", "Captain",
                       "Lieutenant", "Sergeant", "SCOUT", "SAPPER", "SPY","BOMB"]
        for name in piece_priority:
            quantity = self.soldiers[player][name]["Quantity"]
            for _ in range(quantity):
                if positions:
                    row, col = positions.pop()
                    self.board[row, col] = f"{name}_{player}"
                    self.pieces[player][name]["quantity"] += 1
        self.swap_rows([0, 3])
        self.swap_rows([1, 2])
        self.swap_rows([2, 3])

    def swap_rows(self, row_indices):
        row1, row2 = row_indices
        self.board[row1], self.board[row2] = np.copy(self.board[row2]), np.copy(self.board[row1])

    def make_move(self, from_pos, to_pos):
        attacker_piece = self.board[from_pos]
        defender_piece = self.board[to_pos] if self.board[to_pos] != "EMPTY" else None
        attacker_color = "red" if attacker_piece in self.soldiers["red"] else "blue"
        defender_color = "red" if defender_piece in self.soldiers["red"] else "blue"
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.board[from_row][from_col]
        if not self.is_legal_move(from_pos, to_pos):
            raise ValueError("Illegal move.")
        if self.board[to_row][to_col] != "EMPTY":
            attack_result = self.attack(from_pos, to_pos)
            attacker_won = attack_result in ("wins", "Spy defeats Marshal")
            self.update_probabilities_after_move(from_pos, to_pos, attacker_won, attacker_piece, defender_piece, attacker_color, defender_color)
            if attack_result in ("wins", "Spy defeats Marshal"):
                self.board[to_row][to_col] = piece
                self.board[from_row][from_col] = "EMPTY"
            elif attack_result in("loses", "Bomb wins"):
                self.board[from_row][from_col] = "EMPTY"
            elif attack_result == "Draw":
                self.board[from_row][from_col] = "EMPTY"
                self.board[to_row][to_col] = "EMPTY"
            elif attack_result == "game over":
                self.game_over = True
                return
        else:
            self.board[to_row][to_col] = piece
            self.board[from_row][from_col] = "EMPTY"
        self.history.append((from_pos, to_pos))
        self.switch_turn()

    def update_probabilities_after_move(self, attacker_pos, defender_pos, attacker_won, attacker_piece, defender_piece, attacker_color, defender_color):
        if attacker_won:
            if defender_piece in self.soldiers[defender_color]:
                self.soldiers[defender_color][defender_piece]["Quantity"] -= 1
                self.total_pieces[defender_color] -= 1
                if self.soldiers[defender_color][defender_piece]["Quantity"] == 0:
                    del self.soldiers[defender_color][defender_piece]
            for piece in self.prob_board_red if defender_color == "red" else self.prob_board_blue:
                self.prob_board_red[defender_pos][piece] = 1.0 if piece == defender_piece else 0.0
        else:
            for piece in self.prob_board_red if attacker_color == "red" else self.prob_board_blue:
                if piece == attacker_piece:
                    self.prob_board_red[attacker_pos][piece] = 0.0
                    self.prob_board_red[defender_pos][piece] = 1.0
        self.normalize_probabilities()

    def unmake_move(self, move):
        """Undo the last move."""
        if not self.history:
            raise ValueError("No moves to undo.")
        from_pos, to_pos, captured_piece = self.history.pop()
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        self.board[from_row][from_col] = self.board[to_row][to_col]
        self.board[to_row][to_col] = captured_piece
        self.switch_turn()

    def clone(self):
        cloned_game = copy.deepcopy(self)
        cloned_game.board = np.copy(self.board)
        return cloned_game

    def attack(self, from_pos, to_pos):
        attacker = self.board[from_pos[0]][from_pos[1]].split("_")[0]
        defender = self.board[to_pos[0]][to_pos[1]].split("_")[0]
        attacker_color = self.board[from_pos[0]][from_pos[1]].split("_")[1]
        defender_color = self.board[to_pos[0]][to_pos[1]].split("_")[1]
        if attacker_color == defender_color:
            return "Invalid attack: Same team."
        attacker_rank = self.soldiers[attacker_color][attacker]["Rank"]
        defender_rank = self.soldiers[defender_color][defender]["Rank"]
        #self.update_probabilities_after_attack(to_pos, defender, defender_color)
        if attacker_rank == 1 and defender_rank == 10:
            return "Spy defeats Marshal"
        if defender == "BOMB" and attacker != "SAPPER":
            return "Bomb wins"
        if defender == "FLAG":
            return "game over"
        if attacker_rank > defender_rank:
            return "wins"
        elif attacker_rank < defender_rank:
            return "loses"
        else:
            return "Draw"

    def update_probabilities_after_attack(self, pos, revealed_piece, revealed_color):
        if revealed_color == "red":
            self.prob_board_blue[pos] = {p: 1 if p == revealed_piece else 0 for p in self.soldiers["red"].keys()}
        else:
            self.prob_board_red[pos] = {p: 1 if p == revealed_piece else 0 for p in self.soldiers["blue"].keys()}

    def is_legal_move(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.board[from_row][from_col]
        if piece == "EMPTY" or piece.split("_")[0] in ["FLAG", "BOMB"]:
            return False
        if (from_row, from_col) in self.water or (to_row, to_col) in self.water:
            return False
        if not (0 <= to_row < 10 and 0 <= to_col < 10):
            return False
        if self.board[to_row][to_col] != "EMPTY" and self.board[to_row][to_col].split("_")[1] == self.turn:
            return False
        if self.is_repetitive_move(from_pos, to_pos):
            return False
        if piece.split("_")[0] == "SCOUT":
            if from_row == to_row:
                step = 1 if to_col > from_col else -1
                for col in range(from_col + step, to_col, step):
                    if self.board[from_row][col] != "EMPTY":
                        return False
            elif from_col == to_col:
                step = 1 if to_row > from_row else -1
                for row in range(from_row + step, to_row, step):
                    if self.board[row][from_col] != "EMPTY":
                        return False
            else:
                return False
            return True
        return abs(from_row - to_row) + abs(from_col - to_col) == 1

    def is_repetitive_move(self, from_pos, to_pos):
        if len(self.history) < 6:
            return False
        last_moves = self.history[-6:]
        if (to_pos, from_pos) in [last_moves[1], last_moves[3], last_moves[5]]:
            return True
        return False

    def legal_moves(self):
        """Calculate legal moves ensuring all moves are valid."""
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for from_row in range(10):
            for from_col in range(10):
                piece = self.board[from_row, from_col]
                if piece != "EMPTY" and piece.split("_")[1] == self.turn:
                    for d_row, d_col in directions:
                        to_row, to_col = from_row + d_row, from_col + d_col
                        if (0 <= to_row < 10 and 0 <= to_col < 10 and
                                self.is_legal_move((from_row, from_col), (to_row, to_col))):
                            moves.append(((from_row, from_col), (to_row, to_col)))
        return moves

    def switch_turn(self):
        self.turn = "blue" if self.turn == "red" else "red"

    def status(self):
        """Return the status of the game."""
        red_flag = any("FLAG_red" in cell for row in self.board for cell in row)
        blue_flag = any("FLAG_blue" in cell for row in self.board for cell in row)
        if not red_flag:
            return "blue wins"
        if not blue_flag:
            return "red wins"
        if not any("SCOUT_red" in cell or "SCOUT_blue" in cell for row in self.board for cell in row):
            return "draw"
        return "ongoing"

    def encode(self):
        """Encode the board state into a NumPy vector for faster processing."""
        vector = np.zeros((10, 10, 12), dtype=np.int8)
        piece_to_index = {
            name: idx for idx, name in enumerate(self.soldiers["red"].keys())
        }
        for row in range(10):
            for col in range(10):
                cell = self.board[row, col]
                if cell != "EMPTY":
                    piece, color = cell.split("_")
                    vector[row, col, piece_to_index[piece]] = 1
        turn_vector = np.array([1 if self.turn == "red" else 0], dtype=np.int8)
        red_counts = np.array(
            [self.soldiers["red"][p]["Quantity"] - self.pieces["red"][p]["quantity"]
            for p in self.soldiers["red"]], dtype=np.int8)
        blue_counts = np.array(
            [self.soldiers["blue"][p]["Quantity"] - self.pieces["blue"][p]["quantity"]
            for p in self.soldiers["blue"]], dtype=np.int8)
        last_moves = np.zeros((5, 2, 2), dtype=np.int8)
        for i, move in enumerate(self.history[-5:]):
            last_moves[i] = move
        legal_moves_matrix = np.zeros((10, 10, 4), dtype=np.int8)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for row in range(10):
            for col in range(10):
                if self.board[row, col] != "EMPTY" and self.board[row, col].split("_")[1] == self.turn:
                    for d_idx, (dr, dc) in enumerate(directions):
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 10 and 0 <= new_col < 10 and self.is_legal_move((row, col), (new_row, new_col)):
                            legal_moves_matrix[row, col, d_idx] = 1
        prob_board = self.prob_board_red if self.turn == "red" else self.prob_board_blue
        piece_list = list(self.soldiers["red"].keys())
        prob_vector = np.array([
            [
                [prob_board.get((r, c), {}).get(p, 0.0) for p in piece_list]
                for c in range(10)
            ]
            for r in range(10)
        ], dtype=np.float32)

        static_pieces = np.zeros((10, 10, 1), dtype=np.int8)
        for row in range(10):
            for col in range(10):
                if self.board[row, col] in ["FLAG_red", "BOMB_red", "FLAG_blue", "BOMB_blue"]:
                    static_pieces[row, col, 0] = 1
        danger_map = np.zeros((10, 10, 1), dtype=np.int8)
        for move in self.history:
            _, to_pos = move
            danger_map[to_pos] += 1
        full_vector = np.concatenate((
            vector.flatten(), # 10x10 matrix with 12 channels for storing tool information
            turn_vector, # Current turn (1 if red, 0 if blue)
            red_counts, # The amount of pieces left for the red player
            blue_counts, # The amount of pieces left for the blue player
            last_moves.flatten(), # The last five moves
            legal_moves_matrix.flatten(), # A matrix for storing legal moves for each square
            prob_vector.flatten(), # Probability matrix for each slot
            static_pieces.flatten(), # A matrix indicating static tools such as flags and bombs
            danger_map.flatten() # A matrix marking "dangerous" slots where recent attacks have occurred
        ))
        return full_vector

    def decode(self, action_index):
        """Decode an action index into a move."""
        total_positions = 100  # 10x10 board
        from_index = action_index // total_positions
        to_index = action_index % total_positions
        from_pos = (from_index // 10, from_index % 10)
        to_pos = (to_index // 10, to_index % 10)
        return from_pos, to_pos

    def __str__(self):
        """Visualize the board."""
        board_representation = "    " + "  |  ".join(map(str, range(10))) + "\n"
        board_representation += "   " + "-" * 31 + "\n"
        for row_idx in range(10):
            row_str = f"{row_idx:2} | " + " | ".join(
                [self._get_display_piece(self.board[row_idx, col_idx], row_idx, col_idx) for col_idx in range(10)]
            )
            board_representation += row_str + "\n" + "   " + "-" * 31 + "\n"
        return board_representation

    def _get_display_piece(self, piece, row_idx, col_idx):
        """Display each piece or water tile."""
        if (row_idx, col_idx) in self.water:
            return "WAT"
        if piece == "EMPTY":
            return "---"
        if ("red" in piece and self.turn == "blue") or ("blue" in piece and self.turn == "red"):
            return "RED" if self.turn == "blue" else "BLUE"
        return piece.split("_")[0][:3].upper()

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = move

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.legal_moves())

    def best_child(self, exploration_weight=1.41):
        if not self.children:
            return None
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                continue
            uct_value = (child.wins / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child

class MCTSPlayer:
    def __init__(self, simulations=1000, exploration_weight=1.41):
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        
    def choose_best_move_based_on_probabilities(self, game, legal_moves):
        prob_board = game.get_prob_board()
        best_move = None
        best_score = -float("inf")
        
        for move in legal_moves:
            from_pos, to_pos = move
            prob_distribution = prob_board.get(to_pos, {})
            
            attacker_piece = game.board[from_pos[0]][from_pos[1]]
            if attacker_piece == "EMPTY" or "_" not in attacker_piece:
                continue
            
            attacker_piece = attacker_piece.split("_")[0]
            if attacker_piece not in game.soldiers[game.turn]:
                continue
            
            attacker_rank = game.soldiers[game.turn][attacker_piece]["Rank"]
            attack_score = 0

            for p in prob_distribution:
                if p not in game.soldiers[game.turn]:
                    continue
                defender_rank = game.soldiers[game.turn][p]["Rank"]
                if attacker_rank >= defender_rank:
                    attack_score += prob_distribution[p] * defender_rank
            attack_score += np.random.uniform(0, 0.1)
            if attack_score > best_score:
                best_score = attack_score
                best_move = move
        return best_move

    def choose_move(self, game):
        legal_moves = game.legal_moves()
        best_prob_move = self.choose_best_move_based_on_probabilities(game, legal_moves)
        if best_prob_move:
            return best_prob_move
        root = MCTSNode(game.clone())
        for _ in range(self.simulations):
            if _ % 100 == 0:
                print(f"🔄 Running MCTS Simulation {_}/{self.simulations}...")
            node = self._select(root)
            if not node:
                continue
            result = self._simulate(node.state)
            self._backpropagate(node, result)
        best_child = root.best_child(exploration_weight=0)
        if best_child:
            return best_child.move
        return None

    def _select(self, node):
        while node.children and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                return node
            node = best_child
        return self._expand(node)

    def _expand(self, node):
        legal_moves = node.state.legal_moves()
        tried_moves = {child.move for child in node.children}
        for move in legal_moves:
            if move not in tried_moves:
                new_state = node.state.clone()
                new_state.make_move(*move)
                child = MCTSNode(new_state, parent=node, move=move)
                node.add_child(child)
                return child
        return None

    def _simulate(self, game, max_depth = 70):
        depth = 0
        while game.status() == "ongoing" and depth < max_depth:
            legal_moves = game.legal_moves()
            if not legal_moves:
                # print(f"⚠️ No valid move found for {game.turn}. Ending game as a draw.")
                # game.game_over = True
                break
            move = random.choice(legal_moves)
            game.make_move(*move)
            depth += 1
        return game.status()

    def _backpropagate(self, node, result):
        while node:
            node.visits += 1
            if result == node.state.turn:
                node.wins += 1
            elif result == "draw":
                node.wins += 0.5
            node = node.parent

class GameNetwork(nn.Module):
    def __init__(self, input_dim, policy_output_dim, value_output_dim=1, hidden_dim=256):
        """
        Neural network for predicting policy (best move) and value (win probability)
        :param input_dim: Size of input feature vector (output of `encode`)
        :param policy_output_dim: Number of possible actions (output of `decode`)
        :param value_output_dim: Output size of the value head (1 for win probability)
        :param hidden_dim: Number of neurons in hidden layers
        """
        super(GameNetwork, self).__init__()

        # Common feature extractor (shared layers)
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
        """ Evaluate a leaf node using the neural network. return: Value estimate (-1 to 1) """
        state_tensor = torch.tensor(node.state.encode(), dtype=torch.float32).unsqueeze(0)
        _, value_estimate = self.network(state_tensor)
        return value_estimate.item()

    def _backpropagate(self, node, value):
        """ Backpropagate the value estimate up the tree """
        while node:
            node.update(value)
            node = node.parent

class PreTrain:
    def __init__(self, num_games=10000, filename="games_data.json"):
        self.num_games = num_games
        self.filename = filename 

    def generate_self_play_games(self):
        games = []
        for _ in range(self.num_games):
            game = Stratego()
            print(f"🔄 Generating game {len(games) + 1}/{self.num_games}...")
            mcts_player = MCTSPlayer(simulations=700, exploration_weight=2)
            game.auto_place_pieces_for_player("red")  # Automatically place pieces
            game.auto_place_pieces_for_player("blue")  # Automatically place pieces
            game_history = []
            i = 1
            MAX_MOVES = 700
            while not game.game_over:
                if game.turn == "red":  # MCTS player (red)
                    move = mcts_player.choose_move(game)
                    if move is None:
                        print("⚠️ No valid move found. Skipping turn.")
                        break 
                    game.make_move(*move)
                else:  # MCTS player (blue)
                    move = mcts_player.choose_move(game)
                    if move is None:
                        print("⚠️ No valid move found. Skipping turn.")
                        break 
                    game.make_move(*move)
                game_history.append(game.encode())  # Store the game state
                if i >= MAX_MOVES:
                    print("🏁 Draw - you have reached the move limit!")
                    game.game_over = True
                i = i+1
            games.append((game_history, game.status()))  # Record game history and outcome
        return games

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

    def save_games_data(self, games_data):
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
        games_data_serializable = convert_to_serializable(games_data)  
        with open(self.filename, "w") as f:
            json.dump(games_data_serializable, f)
        print(f"✅ Games data saved to {self.filename}")

    def load_games_data(self):
        try:
            if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
                print("❌ No saved games found or file is empty, generating new games...")
                return None
            with open(self.filename, "r") as f:
                games_data = json.load(f)
            if not games_data:  # בודק אם הקובץ ריק
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

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    # Initialize pre-training and generate self-play games
    pretrain = PreTrain(num_games=10000)  # Reduced for quick testing
    games_data = pretrain.load_games_data()
    if not games_data:
        games_data = pretrain.generate_self_play_games()
        pretrain.save_games_data(games_data)
    if not games_data:
        print("❌ Error: No game data generated.")
        return
    inputs, policy_labels, value_labels = pretrain.prepare_training_data(games_data)
    if inputs is None or policy_labels is None or value_labels is None:
        print("❌ Error: Training data was not prepared correctly.")
        return
    game = Stratego()
    print("🛠 Automatically placing pieces for both players...")
    game.auto_place_pieces_for_player("blue")
    game.auto_place_pieces_for_player("red")
    legal_moves = game.legal_moves()
    if not legal_moves:
        print("❌ Error: No legal moves available at the start of the game.")
        return
    network = GameNetwork(input_dim=inputs.shape[1], policy_output_dim=len(policy_labels[0]) , value_output_dim=1)
    trainer = Train(network, inputs, policy_labels, value_labels, epochs=10, batch_size=32)
    trainer.train()
    network.load_model("trained_game_network.pth")
    puct_player = PUCTPlayer(network, simulations=500, cpuct=1.5)
    print("🎲 Initial game board:")
    print(game)
    while not game.game_over:
        print(f"\n🎮 It's {game.turn}'s turn!")
        print(game)
        if game.turn == "red":
            try:
                from_row = int(input("🔹 Enter row to move from (0-9): "))
                from_col = int(input("🔹 Enter column to move from (0-9): "))
                to_row = int(input("🔹 Enter row to move to (0-9): "))
                to_col = int(input("🔹 Enter column to move to (0-9): "))
                game.make_move((from_row, from_col), (to_row, to_col))
            except ValueError as e:
                print(f"⚠️ Error: {e}")
                continue
        else:
            print("🤖 PUCT player is thinking...")
            try:
                move = puct_player.choose_move(game)
                if move:
                    game.make_move(*move)
                    print(f"🤖 PUCT player moved from {move[0]} to {move[1]}")
                else:
                    print("⚠️ PUCT failed to find a move.")
                    break
            except Exception as e:
                print(f"⚠️ PUCT Error: {e}")
                break
        status = game.status()
        if status != "ongoing":
            print(f"🏁 Game Over! {status}")
            break
    print("🏁 Game Over!")

if __name__ == "__main__":
    main()
