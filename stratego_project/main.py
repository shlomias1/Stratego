from stratego import Stratego
from mcts import MCTSPlayer
from puct import PUCTPlayer
from game_net import GameNetwork
from training import PreTrain, Train

def main():
    network = GameNetwork(input_dim=inputs.shape[1], policy_output_dim=len(game.legal_moves()), value_output_dim=1)
    pretrain = PreTrain(num_games=10000)
    games_data = pretrain.generate_self_play_games() 
    inputs, policy_labels, value_labels = pretrain.prepare_training_data(games_data)  
    trainer = Train(network, inputs, policy_labels, value_labels, epochs=10, batch_size=32)
    trainer.train() 
    network.load_model("trained_game_network.pth")
    puct_player = PUCTPlayer(network, simulations=800, cpuct=1.5)
    game = Stratego()
    print("Auto-placing pieces for both players...")
    game.auto_place_pieces_for_player("blue")  
    game.auto_place_pieces_for_player("red")  
    # game.input_pieces('red')
    print("Starting board:")
    print(game)
    while not game.game_over:
        print(f"\nIt's {game.turn}'s turn!")
        print(game)
        if game.turn == "red":
            try:
                from_row = int(input("Enter the row to move from (0-9): "))
                from_col = int(input("Enter the column to move from (0-9): "))
                to_row = int(input("Enter the row to move to (0-9): "))
                to_col = int(input("Enter the column to move to (0-9): "))
                game.make_move((from_row, from_col), (to_row, to_col))
            except ValueError as e:
                print(f"Error: {e}")
                continue
        else: 
            print("PUCT player is thinking...")
            try:
                move = puct_player.choose_move(game)  
                game.make_move(*move)
                print(f"PUCT player moved from {move[0]} to {move[1]}")
            except Exception as e:
                print(f"PUCT Error: {e}")
                break
        status = game.status()
        if status != "ongoing":
            print(f"Game Over! {status}")
            break
    print("Game Over!")

if __name__ == "__main__":
    main()
