import os
from datetime import datetime


class PGNSaver:
    def __init__(self, event="Live Game", site="Local", white="White", black="Black"):
        # Create a folder for PGN files if not existing
        base_dir = os.path.dirname(__file__)
        self.logs_dir = os.path.join(base_dir, "game_logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        now = datetime.now()
        date_str = now.strftime("%Y.%m.%d")
        time_str = now.strftime("%H%M%S")
        filename = f"{white}_{black}_{date_str}_{time_str}.pgn"
        self.filepath = os.path.join(self.logs_dir, filename)
        self.headers = (
            f'[Event "{event}"]\n'
            f'[Site "{site}"]\n'
            f'[Date "{date_str}"]\n'
            f'[Round "-"]\n'
            f'[White "{white}"]\n'
            f'[Black "{black}"]\n'
            f'[Result "*"]\n\n'
        )
        self.current_pgn = ""
        self.last_main_line = []  # store last full move list for branching

    def format_move_line(self, move_list):
        """Format move_list (list of Move objects) into PGN moves.
        If an undo occurred, detect branch divergence (simplified)."""
        pgn = ""
        # If new move log is not a prefix of the previous, add variation markers at divergence.
        divergence = 0
        for i, mv in enumerate(move_list):
            if i < len(self.last_main_line) and mv.move_id == self.last_main_line[i]:
                divergence = i + 1
            else:
                break
        # Write main line moves up to divergence unchanged.
        tokens = []
        for i, mv in enumerate(move_list):
            # standard PGN numbering: pair one white move and optionally black move.
            if i % 2 == 0:
                tokens.append(f"{i//2+1}.")
            # Here we assume str(mv) returns standard algebraic notation.
            tokens.append(str(mv))
        main_line = " ".join(tokens)
        # If the new move list is shorter than the previous, add a variation marker showing the offshoot.
        if len(move_list) < len(self.last_main_line):
            # Mark branch end with "( ... )" containing the abandoned line.
            offshoot = " ".join(str(mv) for mv in self.last_main_line[len(move_list) :])
            main_line += f" ( {offshoot} )"
        self.last_main_line = [mv.move_id for mv in move_list]
        return main_line

    def update_pgn(self, move_list):
        """Update the file with the current PGN line."""
        main_line = self.format_move_line(move_list)
        self.current_pgn = self.headers + main_line
        with open(self.filepath, "w") as f:
            f.write(self.current_pgn)
