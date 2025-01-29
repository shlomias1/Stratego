def get_display_piece(piece, row_idx, col_idx, water, turn):
    """Display each piece or water tile."""
    if (row_idx, col_idx) in water:
        return "WAT"
    if piece == "EMPTY":
        return "---"
    if ("red" in piece and turn == "blue") or ("blue" in piece and turn == "red"):
        return "RED" if turn == "blue" else "BLUE"
    return piece.split("_")[0][:3].upper()
