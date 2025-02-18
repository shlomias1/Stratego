import numpy as np
import copy
import re
import random
import math
from utils import _get_display_piece

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
        vector = np.zeros((10, 10, 12), dtype=np.uint8)
        piece_to_index = {
            name: idx for idx, name in enumerate(self.soldiers["red"].keys())
        }
        for row in range(10):
            for col in range(10):
                cell = self.board[row, col]
                if cell != "EMPTY":
                    piece, color = cell.split("_")
                    vector[row, col, piece_to_index[piece]] = 1
        turn_vector = np.array([1 if self.turn == "red" else 0], dtype=np.uint8)
        red_counts = np.array(
            [self.soldiers["red"][p]["Quantity"] - self.pieces["red"][p]["quantity"]
            for p in self.soldiers["red"]], dtype=np.uint8)
        blue_counts = np.array(
            [self.soldiers["blue"][p]["Quantity"] - self.pieces["blue"][p]["quantity"]
            for p in self.soldiers["blue"]], dtype=np.uint8)
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
        ], dtype=np.float16)

        static_pieces = np.zeros((10, 10, 1), dtype=np.int8)
        for row in range(10):
            for col in range(10):
                if self.board[row, col] in ["FLAG_red", "BOMB_red", "FLAG_blue", "BOMB_blue"]:
                    static_pieces[row, col, 0] = 1
        # danger_map = np.zeros((10, 10, 1), dtype=np.int8)
        # for move in self.history:
        #     _, to_pos = move
        #     danger_map[to_pos] += 1
        full_vector = np.concatenate((
            vector.flatten(), # 10x10 matrix with 12 channels for storing tool information
            turn_vector, # Current turn (1 if red, 0 if blue)
            red_counts, # The amount of pieces left for the red player
            blue_counts, # The amount of pieces left for the blue player
            last_moves.flatten(), # The last five moves
            legal_moves_matrix.flatten(), # A matrix for storing legal moves for each square
            prob_vector.flatten(), # Probability matrix for each slot
            static_pieces.flatten() # A matrix indicating static tools such as flags and bombs
            #danger_map.flatten() # A matrix marking "dangerous" slots where recent attacks have occurred
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
                [_get_display_piece(self.board[row_idx, col_idx], row_idx, col_idx) for col_idx in range(10)]
            )
            board_representation += row_str + "\n" + "   " + "-" * 31 + "\n"
        return board_representation
