from stratego import Stratego
from mcts import MCTSPlayer

def main():
    game = Stratego()
    mcts_player = MCTSPlayer(simulations=100, exploration_weight=2)
    print("Auto-placing pieces for both players...")
    game.auto_place_pieces()
    # game.input_pieces('red')
    print("Starting board:")
    print(game)
    while not game.game_over:
        print(f"\nIt's {game.turn}'s turn!")
        print(game)
        if game.turn == "red":  # Human player
            try:
                from_row = int(input("Enter the row to move from (0-9): "))
                from_col = int(input("Enter the column to move from (0-9): "))
                to_row = int(input("Enter the row to move to (0-9): "))
                to_col = int(input("Enter the column to move to (0-9): "))
                game.make_move((from_row, from_col), (to_row, to_col))
            except ValueError as e:
                print(f"Error: {e}")
                continue
        else:  # MCTS player
            print("MCTS player is thinking...")
            try:
                move = mcts_player.choose_move(game)
                game.make_move(*move)
                print(f"MCTS player moved from {move[0]} to {move[1]}")
            except Exception as e:
                print(f"MCTS Error: {e}")
                break
        # Check game status after each move
        status = game.status()
        if status != "ongoing":
            print(f"Game Over! {status}")
            break
    print("Game Over!")

if __name__ == "__main__":
    main()