# Stratego Game

This project implements a modular version of the classic board game **Stratego** in Python. It allows players to set up their pieces, take turns moving them, and engage in battles based on predefined rules. The game supports both automatic and manual piece placement, and it includes an AI opponent based on **Monte Carlo Tree Search (MCTS)**.

## Features

### Game Board
- A **10x10** board is used to play the game.
- Includes **water tiles** that cannot be occupied.

### Players
- Two players: **Red** and **Blue**.
- Each player has a predefined set of pieces, including **Flag, Bomb, Spy, Scout, Sapper, Sergeant, Lieutenant, Captain, Major, Colonel, General, and Marshal**.
- Each piece has a **rank** and **quantity** according to Stratego rules.

### Gameplay Mechanics
- **Movement**: Pieces move according to the rules of Stratego.
- **Attacking**: Battles are resolved based on rank.
- **Victory Conditions**: A player wins by capturing the opponent's flag or eliminating all movable pieces.
- **AI Opponent**: Uses **Monte Carlo Tree Search (MCTS)** for decision-making.

### Modes
- **Automatic piece placement**: The game automatically places all pieces in designated rows.
- **Manual piece placement**: Players can manually place their pieces with input validation.

## Modules and Key Classes

The game is structured using multiple modules for clarity and maintainability.

### `stratego.py`
Handles core game logic, including:
- `Stratego` class (game management, moves, attack resolution, turn switching, etc.).
- Board setup and piece placement functions.
- Move validation and victory conditions【27†source】.

### `mcts.py`
Implements the **Monte Carlo Tree Search (MCTS) AI**, including:
- `MCTSPlayer` class for AI decision-making.
- `MCTSNode` class for tree search and move simulation【26†source】.

### `utils.py`
Contains helper functions, such as:
- `get_display_piece()` for board visualization【28†source】.

### `main.py`
Entry point for the game, which:
- Initializes the game and AI.
- Handles user input and gameplay loop【25†source】.

## How to Run

### Requirements
Ensure you have Python 3.x installed on your system.

### Running the Game
Save the files in the same directory and run:
```bash
python main.py
```

### Gameplay Instructions
- The game starts with **automatic piece placement**.
- To enable **manual placement**, modify `main.py` and uncomment:
  ```python
  game.input_pieces('red')
  game.input_pieces('blue')
  ```
- Players take turns moving their pieces by entering the **row and column** of the piece to move and its destination.
- The game ends when one player **captures the opponent's flag** or **all movable pieces** of a player are eliminated.

## Rules

### Movement
- Most pieces move **one square** horizontally or vertically.
- The **Scout** can move any number of squares in a straight line.

### Attacking
- Pieces attack by moving into a square occupied by an opponent.
- The **higher-ranked piece** wins the battle.
- If ranks are **equal**, both pieces are removed.
- **Spy** can defeat the **Marshal** if it attacks first.
- **Bombs** cannot move and can only be defused by a **Sapper**.

### Victory Conditions
- **Capture the opponent's Flag**.
- **Eliminate all movable enemy pieces**.

## Example Output
```
It's red's turn!
Enter the row to move from (0-9): 6
Enter the column to move from (0-9): 0
Enter the row to move to (0-9): 5
Enter the column to move to (0-9): 0
Move executed successfully.
```

## Future Improvements
- Add a **graphical interface** for a better user experience.
- Implement a **stronger AI** using deep learning.
- Allow **saving and loading** game states.

## License
This project is open-source and available under the **MIT License**.
