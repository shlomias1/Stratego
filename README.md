# Stratego Game

This project implements a modular version of the classic board game **Stratego** in Python. The game includes both manual and automatic piece placement and supports AI opponents using **Monte Carlo Tree Search (MCTS)** and **PUCT** (Predictor Upper Confidence Bound for Trees). Additionally, a **deep learning-based AI** is integrated using a **neural network** to enhance gameplay decision-making.

## Features

### Game Board
- A **10x10 board** is used to play the game.
- Includes **water tiles** that cannot be occupied.

### Players
- Two players: **Red** and **Blue**.
- Each player has a predefined set of pieces, including **Flag**, **Bomb**, **Spy**, **Scout**, **Sapper**, **Sergeant**, **Lieutenant**, **Captain**, **Major**, **Colonel**, **General**, and **Marshal**.
- Each piece has a **rank** and **quantity** according to Stratego rules.

### Gameplay Mechanics
- **Movement**: Pieces move according to the rules of Stratego.
- **Attacking**: Battles are resolved based on the piece's rank.
- **Victory Conditions**: A player wins by capturing the opponent's flag or eliminating all movable pieces.

### AI Opponent
- **Monte Carlo Tree Search (MCTS)** and **PUCT** (using a neural network for policy/value estimation).
- AI utilizes **probabilistic reasoning** to determine the best moves based on historical moves and piece rankings.

### Modes
- **Automatic piece placement**: The game automatically places all pieces in designated rows.
- **Manual piece placement**: Players can manually place their pieces with input validation.

## Modules and Key Classes

The game is structured using multiple modules for clarity and maintainability:

### **stratego.py**
Handles the core game logic, including:
- **Stratego class**: Manages the game state, moves, attack resolution, and turn switching.
- Board setup, piece placement, and validation of legal moves.
- Probabilistic piece tracking based on observed moves.

### **mcts.py**
Implements the **Monte Carlo Tree Search (MCTS)** AI, including:
- **MCTSPlayer class**: Decision-making for the AI based on MCTS.
- **MCTSNode class**: Tree search and move simulation.

### **puct.py**
Implements the **PUCT** AI, including:
- **PUCTNode class**: Nodes in the PUCT search tree for move decision-making.
- **PUCTPlayer class**: AI decision-making using the PUCT algorithm, based on a neural network.

### **game_net.py**
Defines the **GameNetwork** class, which is a **neural network** used for predicting:
- **Policy** (best move probability).
- **Value** (win probability) for the game state.
- Contains methods to save and load the model.

### **training.py**
Contains the **PreTrain** and **Train** classes:
- **PreTrain**: Generates self-play games using MCTS to collect training data and saves them to `games_data.json`.
- **Train**: Trains the **GameNetwork** using the generated self-play games, logging loss and accuracy per epoch.

### **utils.py**
Contains helper functions such as:
- `_get_display_piece()`: For visualizing the board and pieces, including water tiles.
- `_create_log()`: Logs important game events and model training updates.
- `Connect_CUDA()`: Manages CUDA availability and ensures the neural network runs on the GPU if available.

### **main.py**
The entry point for the game:
- Initializes the game, AI player, and neural network.
- Handles user input and gameplay loop.
- Logs game progress and training updates.

## How to Run

### Requirements
Ensure you have **Python 3.x** installed on your system. Install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Game
Save all the files in the same directory and run the following command:
```bash
python main.py
```

### Training the Neural Network
To train the AI using self-play games, run:
```bash
python main.py --train
```

## Gameplay Instructions
- The game starts with **automatic piece placement**.
- To enable **manual piece placement**, edit `main.py` and uncomment the following lines:
```python
game.input_pieces('red')
game.input_pieces('blue')
```
- Players take turns moving their pieces by entering the row and column of the piece to move and its destination.
- The game ends when one player captures the opponent's flag or all movable pieces of a player are eliminated.

## Rules

### Movement
- Most pieces move one square horizontally or vertically.
- The **Scout** can move any number of squares in a straight line.

### Attacking
- Pieces attack by moving into a square occupied by an opponent.
- The higher-ranked piece wins the battle.
- If ranks are equal, both pieces are removed.
- **Spy** can defeat the **Marshal** if it attacks first.
- **Bombs** cannot move and can only be defused by a **Sapper**.

### Victory Conditions
- **Capture the opponent's Flag**.
- **Eliminate all movable enemy pieces**.

## Example Output
```bash
It's red's turn!
Enter the row to move from (0-9): 6
Enter the column to move from (0-9): 0
Enter the row to move to (0-9): 5
Enter the column to move to (0-9): 0
Move executed successfully.
```

## Logging and Debugging
- Training logs are saved in **`training_log.txt`**.
- Game generation logs are stored in **`game_generation_log.txt`**.
- General logs are written to **`logs.txt`**.

To monitor training and memory usage in real-time:
```bash
tail -f training_log.txt
```

## Future Improvements
- Add a **graphical interface** for a better user experience.
- Improve **AI efficiency** with advanced deep learning techniques.
- Implement **game state saving and loading**.

## License
This project is open-source and available under the **MIT License**.
