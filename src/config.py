# Game board settings
BOARD_SIZE = (10, 10)
MAX_MOVES = 1500
HISTORY_SIZE = 6
# Neural network training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50
HIDDEN_LAYER_SIZE = 256
FEATURE_VECTOR_SIZE = 2421
# Parameters for Genroot Games
NUM_GAMES = 10000
# Parameters for MCTS and PUCT
MCTS_SIMULATIONS = 1000
PUCT_SIMULATIONS = 1000
EXPLORATION_WEIGHT = 1
MAX_DEPTH = 100
RESULT_MAP = {
    "red wins": 1,    # Red Victory → 1
    "blue wins": -1,  # Blue Victory → -1
    "draw": 0.5         # Draw → 0
}
#dirs
DATA_DIR = "data"
LOG_DIR = "logs"
