from datetime import datetime
import torch
import numpy as np

def _get_display_piece(piece, row_idx, col_idx, water, turn):
    """Display each piece or water tile."""
    if (row_idx, col_idx) in water:
        return "WAT"
    if piece == "EMPTY":
        return "---"
    if ("red" in piece and turn == "blue") or ("blue" in piece and turn == "red"):
        return "RED" if turn == "blue" else "BLUE"
    return piece.split("_")[0][:3].upper()

def _create_log(log_msg, log_type, log_file = "logs.txt"):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as log:
        log.write(f'{log_type} : {log_msg} | {current_time} \n')

def Connect_CUDA():
    log_type = "Warning" 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # גודל זיכרון ב-GB
        log_msg1 = f"✅ Running on GPU: {gpu_name} | Memory: {gpu_memory:.2f} GB"
        log_msg2 = f"🔍 GPU Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        log_msg3 = f"🔍 GPU Cached Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        _create_log(log_msg1, log_type)
        _create_log(log_msg2, log_type)
        _create_log(log_msg3, log_type)
        print(log_msg1)
        print(log_msg2)
        print(log_msg3)
    else:
        device = torch.device("cpu")
        log_msg = "Running on CPU - No GPU detected!"
        _create_log(log_msg, log_type)
        print(log_msg)
    return device

def swap_rows(board, row_indices):
    row1, row2 = row_indices
    board[row1], board[row2] = np.copy(board[row2]), np.copy(board[row1])
