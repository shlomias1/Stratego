from datetime import datetime
import torch

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
        log_msg = f"Running on GPU: {torch.cuda.get_device_name(0)}"
        _create_log(log_msg, log_type)
        print(log_msg)
    else:
        device = torch.device("cpu")
        log_msg = "Running on CPU"
        _create_log(log_msg, log_type)
        print(log_msg)
