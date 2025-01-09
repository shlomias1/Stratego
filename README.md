# Stratego Game

This project implements a simplified version of the classic board game **Stratego** in Python. It allows players to set up their pieces, take turns moving them, and engage in battles based on predefined rules. The game includes automatic setup and manual piece placement modes, and it supports basic gameplay mechanics such as movement, attacking, and victory conditions.

## Features

- **Game Board**:
  - A 10x10 board is used to play the game.
  - Includes water tiles that cannot be occupied.
- **Players**:
  - Two players: `red` and `blue`.
  - Each player has a predefined set of pieces, including `Flag`, `Bomb`, `Spy`, and others, each with specific ranks and quantities.
- **Gameplay Mechanics**:
  - Movement and attacking are based on the rules of Stratego.
  - Pieces can only move to valid positions and engage in battles according to their ranks.
  - Victory conditions include capturing the opponent's flag or eliminating all movable pieces.
- **Modes**:
  - Automatic piece placement for both players.
  - Manual piece placement with input validation.

## Classes and Methods

### `Stratego`

#### Initialization (`__init__`)
- Sets up the game board, water tiles, and player pieces.
- Initializes game state variables such as turn and history.

#### `input_pieces(player)`
- Allows manual placement of pieces for the given player.
- Ensures pieces are placed within valid rows and that the maximum quantity is not exceeded.

#### `display_remaining_pieces(player)`
- Displays the remaining pieces available for placement for the given player.

#### `place_piece(player, piece_name, row, col)`
- Places a piece on the board at the specified position.
- Validates placement rules (e.g., no placement on water tiles or occupied spaces).

#### `auto_place_pieces()`
- Automatically places all pieces for both players in their designated rows.

#### `make_move(from_pos, to_pos)`
- Moves a piece from one position to another.
- Handles attacks if the destination is occupied by an opponent's piece.
- Updates the game board and turn.

#### `attack(from_pos, to_pos)`
- Resolves an attack between two pieces based on their ranks.
- Returns the result of the attack (e.g., "wins", "loses", "Draw").

#### `is_legal_move(from_pos, to_pos)`
- Validates whether a move is legal based on the game rules.

#### `switch_turn()`
- Switches the turn between `red` and `blue` players.

#### `__str__()`
- Returns a string representation of the game board for display.

## How to Run

1. Ensure you have Python 3.x installed on your system.
2. Save the code in a file named `stratego.py`.
3. Run the script using the command:
   ```
   python stratego.py
   ```

## Gameplay Instructions

1. The game starts with automatic piece placement. To enable manual placement, uncomment the following lines in the `main()` function:
   ```python
   game.input_pieces('red')
   game.input_pieces('blue')
   ```
2. Players take turns to move their pieces by entering the starting and ending positions (row and column).
3. The game continues until one player captures the opponent's flag or no movable pieces remain.

## Rules

- **Movement**:
  - Most pieces can move one square horizontally or vertically.
  - `Scout` pieces can move any number of squares in a straight line, as long as there are no obstacles.
- **Attacking**:
  - Pieces attack by moving into a square occupied by an opponent's piece.
  - The higher-ranked piece wins the battle. If ranks are equal, both pieces are removed.
  - `Spy` can defeat the `Marshal` if it attacks first.
  - `Bomb` cannot move and can only be defused by a `Sapper`.
- **Victory**:
  - Capture the opponent's `Flag`.
  - Eliminate all movable pieces of the opponent.

## Example Output

```
Player red: Please place your pieces.
You can place your pieces only in rows 6-9.
Remaining pieces for player red:
FLAG: 1 out of 1
BOMB: 6 out of 6
...
Enter the piece name (e.g., Flag, Bomb, Spy): FLAG
Enter the row (6-9): 6
Enter the column (0-9): 0
The piece FLAG was successfully placed at position (6, 0).

Board after placement:
    0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
   -----------------------------------
 6 | FLAG | --- | --- | --- | --- | --- | --- | --- | --- | ---
   -----------------------------------
...

It's red's turn!
Enter the row to move from (0-9): 6
Enter the column to move from (0-9): 0
Enter the row to move to (0-9): 5
Enter the column to move to (0-9): 0
Move executed successfully.
```

## Future Improvements

- Add graphical interface for better user experience.
- Implement more advanced AI for single-player mode.
- Allow saving and loading game states.

## License

This project is open-source and available under the MIT License.
