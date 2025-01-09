import re

class Stratego:
    def __init__(self):
        self.board = [['EMPTY'] * 10 for _ in range(10)]
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

    def auto_place_pieces(self):
        print("Automatically placing all pieces for both players...")

        def place_all(player, rows):
            positions = [(row, col) for row in rows for col in range(10)]
            idx = 0
            for name, data in self.soldiers[player].items():
                for _ in range(data["Quantity"]):
                    if idx >= len(positions):
                        raise ValueError("Not enough positions to place all pieces.")
                    row, col = positions[idx]
                    self.board[row][col] = f"{name}_{player}"
                    self.pieces[player][name]["quantity"] += 1
                    idx += 1

        place_all("red", range(6, 10))
        place_all("blue", range(0, 4))
        print("All pieces have been placed.")

    def make_move(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.board[from_row][from_col]

        if not self.is_legal_move(from_pos, to_pos):
            raise ValueError("Illegal move.")

        if self.board[to_row][to_col] != "EMPTY":
            attack_result = self.attack(from_pos, to_pos)
            print(attack_result)
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
                print("Game Over! The opponent wins!")
                return
        else:
            self.board[to_row][to_col] = piece
            self.board[from_row][from_col] = "EMPTY"

        self.history.append((from_pos, to_pos))
        self.switch_turn()
        print("Move executed successfully.")

    def attack(self, from_pos, to_pos):
        attacker = self.board[from_pos[0]][from_pos[1]].split("_")[0]
        defender = self.board[to_pos[0]][to_pos[1]].split("_")[0]
        attacker_color = self.board[from_pos[0]][from_pos[1]].split("_")[1]
        defender_color = self.board[to_pos[0]][to_pos[1]].split("_")[1]

        if attacker_color == defender_color:
            return "Invalid attack: Same team."

        attacker_rank = self.soldiers[attacker_color][attacker]["Rank"]
        defender_rank = self.soldiers[defender_color][defender]["Rank"]

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
        if piece.split("_")[0] != "SCOUT" and abs(from_row - to_row) + abs(from_col - to_col) != 1:
            return False
        return True

    def switch_turn(self):
        self.turn = "blue" if self.turn == "red" else "red"

    def __str__(self):
        board_representation = "    " + " | ".join(map(str, range(10))) + "\n"
        board_representation += "   " + "-" * 31 + "\n"
        for row_idx, row in enumerate(self.board):
            row_str = f"{row_idx:2} | " + " | ".join(
                [self._get_display_piece(piece, row_idx, col_idx) for col_idx, piece in enumerate(row)]
            )
            board_representation += row_str + "\n" + "   " + "-" * 31 + "\n"
        return board_representation

    def _get_display_piece(self, piece, row_idx, col_idx):
        if (row_idx, col_idx) in self.water:
            return "WAT"
        if piece == "EMPTY":
            return "---"
        if ("red" in piece and self.turn == "blue") or ("blue" in piece and self.turn == "red"):
            return "RED" if self.turn == "blue" else "BLUE"
        return piece.split("_")[0][:3].upper()

def main():
    game = Stratego()
    game.auto_place_pieces()
    # game.input_pieces('red')
    # game.input_pieces('blue')
    print("Starting board:")
    print(game)
    while not game.game_over:
        print(f"\nIt's {game.turn}'s turn!")
        print(game)
        try:
            from_row = int(input("Enter the row to move from (0-9): "))
            from_col = int(input("Enter the column to move from (0-9): "))
            to_row = int(input("Enter the row to move to (0-9): "))
            to_col = int(input("Enter the column to move to (0-9): "))
            game.make_move((from_row, from_col), (to_row, to_col))
        except ValueError as e:
            print(f"Error: {e}")
            continue

    print("Game Over!")

if __name__ == "__main__":
    main()