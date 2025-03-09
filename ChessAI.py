import random
import subprocess

# Basic material values; these can be tuned further.
material_values = {"K": 10000, "Q": 9, "R": 5, "B": 3, "N": 3, "P": 1}


class StockfishEngine:

    def __init__(
        self, stockfish_path="/opt/homebrew/Cellar/stockfish/17/bin/stockfish"
    ):
        self.stockfish_path = stockfish_path
        self.process = subprocess.Popen(
            [self.stockfish_path],
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._init_stockfish()

    def _init_stockfish(self):
        self.send_command("uci")
        self.send_command("isready")
        self.send_command("ucinewgame")

    def send_command(self, command):
        if self.process.stdin:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        else:
            raise RuntimeError("Failed to write to Stockfish process stdin.")

    def get_evaluation(self, fen, depth=5):
        self.send_command("position fen " + fen)
        self.send_command("go depth " + str(depth))
        eval_value = None
        while True:
            if self.process.stdout:
                output = self.process.stdout.readline().strip()
            else:
                raise RuntimeError("Failed to read from Stockfish process stdout.")
            if output.startswith("info depth"):
                parts = output.split()
                if "score" in parts:
                    idx = parts.index("score")
                    if parts[idx + 1] == "cp":
                        eval_value = float(parts[idx + 2]) / 100.0
                    elif parts[idx + 1] == "mate":
                        mate = int(parts[idx + 2])
                        eval_value = 10000 if mate > 0 else -10000
            if output.startswith("bestmove"):
                break
        return eval_value

    def quit(self):
        self.send_command("quit")
        self.process.terminate()


def evaluate_board(game_state):
    """
    Returns an evaluation combining Stockfish's base score and additional adjustments.
    Adjustments include base elemental bonuses and extra modifications if a piece occupies an elemental square,
    plus penalties/bonuses for special effects (explosion, shield).
    """
    # Get Stockfish evaluation
    fen = game_state.to_fen()
    engine = StockfishEngine(
        stockfish_path="/opt/homebrew/Cellar/stockfish/17/bin/stockfish"
    )
    base_eval = engine.get_evaluation(fen) or 0.0
    engine.quit()

    adjustment = 0.0
    # Define bonus multipliers in local dictionaries for clarity
    base_bonus = {"W": 0.2, "I": 0.5, "E": 0.3, "F": -0.5}
    piece_bonus = {"W": 0.05, "I": -0.07, "E": 0.1, "F": -0.1}

    # Use the refactored elementalTiles.tiles, not elemental_squares.
    for (r, c), (element, turns) in game_state.elementalTiles.tiles.items():
        adjustment += base_bonus.get(element, 0.0)
        piece = game_state.board[r][c]
        if piece is not None and hasattr(piece, "piece_type"):
            bonus = piece_bonus.get(element, 0.0)
            if piece.color == "w":
                adjustment += bonus
            else:
                adjustment -= bonus

    # Loop over board once to apply special effects from individual pieces.
    for row in game_state.board:
        for piece in row:
            if piece is not None and hasattr(piece, "piece_type"):
                val = material_values.get(piece.piece_type, 0)
                if piece.will_explode:
                    penalty = 0.3 * val
                    adjustment -= penalty if piece.color == "w" else -penalty
                if piece.shielded:
                    bonus = 0.1 * val
                    adjustment += bonus if piece.color == "w" else -bonus

    return base_eval + adjustment


def move_to_uci(move):
    """
    Converts a Move into its UCI string (e.g., "e2e4").
    Assumes move.get_rank_file(row, col) returns standard chess notation.
    """
    start = move.get_rank_file(move.start_row, move.start_column)
    end = move.get_rank_file(move.end_row, move.end_column)
    return (start + end).lower()


def find_best_move(game_state, valid_moves=None):
    """
    Uses Stockfish to determine the best move for the current position.
    Sends the FEN string, retrieves the best move as a UCI string,
    and returns the matching Move from valid_moves (or game_state.get_valid_moves()).
    """
    if valid_moves is None:
        valid_moves = game_state.get_valid_moves()
    fen = game_state.to_fen()
    engine = StockfishEngine(
        stockfish_path="/opt/homebrew/Cellar/stockfish/17/bin/stockfish"
    )
    engine.send_command("position fen " + fen)
    # Reduced depth for faster response:
    engine.send_command("go depth 9")
    bestmove_str = None
    while True:
        output = engine.process.stdout.readline().strip()
        if output.startswith("bestmove"):
            parts = output.split()
            bestmove_str = parts[1]
            break
    engine.quit()
    for move in valid_moves:
        if move_to_uci(move) == bestmove_str:
            return move
    return random.choice(valid_moves) if valid_moves else None
