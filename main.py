import torch
from stratego import Stratego
from mcts import MCTSPlayer
from puct import PUCTPlayer
from game_net import GameNetwork
from training import PreTrain, Train

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