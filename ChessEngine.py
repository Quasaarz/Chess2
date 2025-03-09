import random
from GameHistorySQL import GameHistorySQL
import pickle
import zlib


class Piece:

    def __init__(self, color, piece_type, position, identifier):
        self.identifier = identifier
        self.color = color
        self.piece_type = piece_type
        self.position = position
        self.shielded = False
        self.frozen_turns = 0
        self.electro_move = False
        self.will_explode = False

    def apply_elemental_effect(self, element):
        """Apply the effect of an elemental square.
        W: Shield remains until piece is captured once.
        I: Freeze for 3 turns.
        E: Allows extra move (recorded as e- move).
        F: Mark for explosion on capture.
        """
        if element == "W":
            self.shielded = True
            print(f"{self.piece_type} at {self.position} is shielded!")
        elif element == "I":
            self.frozen_turns = 3
            print(f"{self.piece_type} at {self.position} is frozen for 3 turns!")
        elif element == "E":
            self.electro_move = True
            print(f"{self.piece_type} at {self.position} can move twice!")
        elif element == "F":
            self.will_explode = True
            print(f"{self.piece_type} at {self.position} will explode on capture!")

    def reset_elemental_effects(self):
        """Reset the elemental effects after they are applied."""
        self.shielded = False
        self.frozen_turns = 0
        self.electro_move = False
        self.will_explode = False

    def get_valid_moves(self, board, pins, en_passant_possible, game_state=None):
        """To be overridden by each specific piece type"""
        pass

    @staticmethod
    def sliding_moves(piece, board, pins, directions):
        moves = []
        opponent = "b" if piece.color == "w" else "w"
        piece_pinned = False
        pin_direction = ()
        for i in range(len(pins) - 1, -1, -1):
            if pins[i][0] == piece.position[0] and pins[i][1] == piece.position[1]:
                piece_pinned = True
                pin_direction = (pins[i][2], pins[i][3])
                if piece.piece_type != "Q":
                    pins.pop(i)
                break
        for d in directions:
            for i in range(1, len(board)):
                end_row = piece.position[0] + d[0] * i
                end_col = piece.position[1] + d[1] * i
                if 0 <= end_row < len(board) and 0 <= end_col < len(board):
                    if (
                        not piece_pinned
                        or pin_direction == d
                        or pin_direction == (-d[0], -d[1])
                    ):
                        end_piece = board[end_row][end_col]
                        if end_piece is None:
                            moves.append(
                                Move(piece.position, (end_row, end_col), board)
                            )
                        elif end_piece.color == opponent:
                            moves.append(
                                Move(piece.position, (end_row, end_col), board)
                            )
                            break
                        else:
                            break
                    else:
                        continue
                else:
                    break
        return moves


class Rook(Piece):
    def __init__(self, color, position, identifier):
        super().__init__(color, "R", position, identifier)

    def get_valid_moves(self, board, pins, en_passant_possible, game_state=None):
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        return Piece.sliding_moves(self, board, pins, directions)


class Bishop(Piece):
    def __init__(self, color, position, identifier):
        super().__init__(color, "B", position, identifier)

    def get_valid_moves(self, board, pins, en_passant_possible, game_state=None):
        directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        return Piece.sliding_moves(self, board, pins, directions)


class Queen(Piece):
    def __init__(self, color, position, identifier):
        super().__init__(color, "Q", position, identifier)

    def get_valid_moves(self, board, pins, en_passant_possible, game_state=None):
        directions = [
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, 1),
            (1, -1),
        ]
        return Piece.sliding_moves(self, board, pins, directions)


class Pawn(Piece):
    def __init__(self, color, position, identifier):
        super().__init__(color, "P", position, identifier)

    def get_valid_moves(self, board, pins, en_passant_possible, game_state=None):
        moves = []
        piece_pinned = False
        pin_direction = ()
        for i in range(len(pins) - 1, -1, -1):
            if pins[i][0] == self.position[0] and pins[i][1] == self.position[1]:
                piece_pinned = True
                pin_direction = (pins[i][2], pins[i][3])
                pins.remove(pins[i])
                break

        if self.color == "w":
            move_amount = -1
            start_row = 6
            back_row = 0
            opponent = "b"
            king_row, king_column = self.position
        else:
            move_amount = 1
            start_row = 1
            back_row = 7
            opponent = "w"
            king_row, king_column = self.position
        pawn_promotion = False

        if board[self.position[0] + move_amount][self.position[1]] is None:
            if not piece_pinned or pin_direction == (move_amount, 0):
                if self.position[0] + move_amount == back_row:
                    pawn_promotion = True
                moves.append(
                    Move(
                        self.position,
                        (self.position[0] + move_amount, self.position[1]),
                        board,
                        pawn_promotion=pawn_promotion,
                    )
                )
                if (
                    self.position[0] == start_row
                    and board[self.position[0] + 2 * move_amount][self.position[1]]
                    is None
                ):
                    moves.append(
                        Move(
                            self.position,
                            (self.position[0] + 2 * move_amount, self.position[1]),
                            board,
                        )
                    )
        if self.position[1] - 1 >= 0:
            if not piece_pinned or pin_direction == (move_amount, -1):
                if (
                    board[self.position[0] + move_amount][self.position[1] - 1]
                    is not None
                    and board[self.position[0] + move_amount][
                        self.position[1] - 1
                    ].color
                    == opponent
                ):
                    if self.position[0] + move_amount == back_row:
                        pawn_promotion = True
                    moves.append(
                        Move(
                            self.position,
                            (self.position[0] + move_amount, self.position[1] - 1),
                            board,
                            pawn_promotion=pawn_promotion,
                        )
                    )
                if (
                    self.position[0] + move_amount,
                    self.position[1] - 1,
                ) == en_passant_possible:
                    moves.append(
                        Move(
                            self.position,
                            (self.position[0] + move_amount, self.position[1] - 1),
                            board,
                            en_passant=True,
                        )
                    )
        if self.position[1] + 1 <= len(board) - 1:
            if not piece_pinned or pin_direction == (move_amount, 1):
                if (
                    board[self.position[0] + move_amount][self.position[1] + 1]
                    is not None
                    and board[self.position[0] + move_amount][
                        self.position[1] + 1
                    ].color
                    == opponent
                ):
                    if self.position[0] + move_amount == back_row:
                        pawn_promotion = True
                    moves.append(
                        Move(
                            self.position,
                            (self.position[0] + move_amount, self.position[1] + 1),
                            board,
                            pawn_promotion=pawn_promotion,
                        )
                    )
                if (
                    self.position[0] + move_amount,
                    self.position[1] + 1,
                ) == en_passant_possible:
                    moves.append(
                        Move(
                            self.position,
                            (self.position[0] + move_amount, self.position[1] + 1),
                            board,
                            en_passant=True,
                        )
                    )
        return moves


class King(Piece):
    def __init__(self, color, position, identifier):
        super().__init__(color, "K", position, identifier)

    def get_valid_moves(self, board, pins, en_passant_possible, game_state):
        if game_state is None:
            return []
        moves = []
        ally = self.color
        original_pos = self.position
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for d in directions:
            end_row = self.position[0] + d[0]
            end_col = self.position[1] + d[1]
            if 0 <= end_row < len(board) and 0 <= end_col < len(board[0]):
                end_piece = board[end_row][end_col]
                if end_piece is None or end_piece.color != ally:
                    orig_loc = (
                        game_state.white_king_location
                        if ally == "w"
                        else game_state.black_king_location
                    )
                    if ally == "w":
                        game_state.white_king_location = (end_row, end_col)
                    else:
                        game_state.black_king_location = (end_row, end_col)
                    in_check, _, _ = game_state.check_for_pins_and_checks()
                    if not in_check:
                        moves.append(Move(original_pos, (end_row, end_col), board))
                    if ally == "w":
                        game_state.white_king_location = orig_loc
                    else:
                        game_state.black_king_location = orig_loc
        # Append castle moves from King-specific method
        moves.extend(self.get_castle_moves(board, game_state, ally))
        return moves

    def get_castle_moves(self, board, game_state, ally):
        """Generate castle moves for the king."""
        castle_moves = []
        row, col = self.position
        if game_state.square_under_attack(row, col, ally):
            return castle_moves
        # King-side castle
        if (ally == "w" and game_state.white_castle_king_side) or (
            ally == "b" and game_state.black_castle_king_side
        ):
            if (
                board[row][col + 1] is None
                and board[row][col + 2] is None
                and not game_state.square_under_attack(row, col + 1, ally)
                and not game_state.square_under_attack(row, col + 2, ally)
            ):
                castle_moves.append(
                    Move((row, col), (row, col + 2), board, castle=True)
                )
        # Queen-side castle
        if (ally == "w" and game_state.white_castle_queen_side) or (
            ally == "b" and game_state.black_castle_queen_side
        ):
            if (
                board[row][col - 1] is None
                and board[row][col - 2] is None
                and board[row][col - 3] is None
                and not game_state.square_under_attack(row, col - 1, ally)
                and not game_state.square_under_attack(row, col - 2, ally)
            ):
                castle_moves.append(
                    Move((row, col), (row, col - 2), board, castle=True)
                )
        return castle_moves


class Knight(Piece):
    def __init__(self, color, position, identifier):
        super().__init__(color, "N", position, identifier)

    def get_valid_moves(self, board, pins, en_passant_possible, game_state=None):
        moves = []
        opponent = "b" if self.color == "w" else "w"
        piece_pinned = False
        for i in range(len(pins) - 1, -1, -1):
            if pins[i][0] == self.position[0] and pins[i][1] == self.position[1]:
                piece_pinned = True
                pins.remove(pins[i])
                break

        directions = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]
        for d in directions:
            end_row = self.position[0] + d[0]
            end_column = self.position[1] + d[1]
            if 0 <= end_row < len(board) and 0 <= end_column < len(board):
                if not piece_pinned:
                    end_piece = board[end_row][end_column]
                    if end_piece is not None and end_piece.color == opponent:
                        moves.append(Move(self.position, (end_row, end_column), board))
                    elif end_piece is None:
                        moves.append(Move(self.position, (end_row, end_column), board))
        return moves


class CastleRights:
    """Data storage of current states of castling rights"""

    def __init__(
        self, white_king_side, black_king_side, white_queen_side, black_queen_side
    ):
        self.white_king_side = white_king_side
        self.black_king_side = black_king_side
        self.white_queen_side = white_queen_side
        self.black_queen_side = black_queen_side


class ElementalTiles:
    def __init__(self):
        self.effects = {
            "W": {
                "name": "Wind",
                "effect": "shield",
                "base_bonus": 0.2,
                "piece_bonus": 0.05,
            },
            "I": {
                "name": "Ice",
                "effect": "freeze",
                "base_bonus": 0.5,
                "piece_bonus": -0.07,
            },
            "E": {
                "name": "Electro",
                "effect": "double_move",
                "base_bonus": 0.3,
                "piece_bonus": 0.1,
            },
            "F": {
                "name": "Fire",
                "effect": "explode",
                "base_bonus": -0.5,
                "piece_bonus": -0.1,
            },
        }
        self.tiles = {}  # Mapping from (row, col) to tuple (element, turns)

    def add_random_tile(self, board):
        empty = [
            (r, c)
            for r in range(8)
            for c in range(8)
            if board[r][c] is None and (r, c) not in self.tiles
        ]
        if empty:
            row, col = random.choice(empty)
            element = random.choice(list(self.effects.keys()))
            self.tiles[(row, col)] = (element, 1)

    def update(self, board):
        # Decrement turn counters and remove expired tiles.
        # Also, if a tile’s square becomes empty after a capture, remove that tile.
        expired = [pos for pos, (_, turns) in self.tiles.items() if turns <= 0]
        for pos in expired:
            del self.tiles[pos]
        for pos in list(self.tiles.keys()):
            element, turns = self.tiles[pos]
            # If the square is occupied by an Ice effect and the occupant is still frozen, skip decrement.
            piece = board[pos[0]][pos[1]]
            if piece is not None:
                if element == "I" and piece.frozen_turns > 0:
                    continue
            # Decrement the tile timer.
            self.tiles[pos] = (element, turns - 1)
            # If, after decrement, the square is now empty, remove the tile immediately.
            if board[pos[0]][pos[1]] is None:
                del self.tiles[pos]

    # NEW functions for individual tile effects:
    def apply_effect(self, piece, element):
        if element == "W":
            self.activate_shield(piece)
        elif element == "I":
            self.activate_freeze(piece)
        elif element == "E":
            self.activate_double_move(piece)
        elif element == "F":
            self.activate_explosion(piece)

    def activate_shield(self, piece):
        piece.shielded = True

    def activate_freeze(self, piece):
        piece.frozen_turns = 3

    def activate_double_move(self, piece):
        piece.electro_move = True

    def activate_explosion(self, piece):
        piece.will_explode = True

    # NEW: Explode a 3x3 area around (row, col)
    def explode_piece(self, board, row, col):
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if 0 <= r < 8 and 0 <= c < 8:
                    board[r][c] = None

    # NEW: Determine if an explosion is safe using provided king locations and pinned pieces.
    def is_explosion_safe(
        self,
        board,
        white_king_location,
        black_king_location,
        explosion_area,
        pins,
        ally,
    ):
        if (
            white_king_location in explosion_area
            or black_king_location in explosion_area
        ):
            return False
        for pin in pins:
            if (pin[0], pin[1]) in explosion_area:
                piece = board[pin[0]][pin[1]]
                if piece is not None and piece.color == ally:
                    return False
        return True

    # NEW: Check enemy queen explosion condition.
    def queen_explosion_condition(
        self, board, white_to_move, white_king_location, black_king_location
    ):
        if white_to_move:
            king_loc = white_king_location
            enemy_color = "b"
        else:
            king_loc = black_king_location
            enemy_color = "w"
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r = king_loc[0] + dr
                c = king_loc[1] + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    piece = board[r][c]
                    if (
                        piece is not None
                        and piece.piece_type == "Q"
                        and piece.color == enemy_color
                        and piece.will_explode
                    ):
                        orig_loc = king_loc
                        # Simulate king moving to capture queen:
                        if white_to_move:
                            test_loc = (r, c)
                        else:
                            test_loc = (r, c)
                        # For simplicity, if such a queen is adjacent, deem condition met.
                        return True
        return False


class GameState:
    """
    Class responsible for storing information about the current state of the game.
    The functions within this class are responsible for how moves are made, undone,
    determining valid moves given the current state, and keeping a move log.
    """

    def __init__(self):
        """
        The board is an 8x8 2d list. Each element has 2 characters.
        1st character represents the colour of the piece (b/w).
        2nd character represents the type of the piece.
        "--" represents an empty space with no piece.
        """

        self.board = [
            [
                Rook("b", (0, 0), "R1"),
                Knight("b", (0, 1), "N1"),
                Bishop("b", (0, 2), "B1"),
                Queen("b", (0, 3), "Q1"),
                King("b", (0, 4), "K1"),
                Bishop("b", (0, 5), "B2"),
                Knight("b", (0, 6), "N2"),
                Rook("b", (0, 7), "R2"),
            ],
            [Pawn("b", (1, i), f"P{i+1}") for i in range(8)],
            [None for _ in range(8)],
            [None for _ in range(8)],
            [None for _ in range(8)],
            [None for _ in range(8)],
            [Pawn("w", (6, i), f"P{i+1}") for i in range(8)],
            [
                Rook("w", (7, 0), "R1"),
                Knight("w", (7, 1), "N1"),
                Bishop("w", (7, 2), "B1"),
                Queen("w", (7, 3), "Q1"),
                King("w", (7, 4), "K1"),
                Bishop("w", (7, 5), "B2"),
                Knight("w", (7, 6), "N2"),
                Rook("w", (7, 7), "R2"),
            ],
        ]

        # Update king locations for 8x8 board
        self.frozen_pieces = {}
        self.white_to_move = True
        self.move_log = []
        # Initialize elemental effects
        self.elementalTiles = ElementalTiles()
        self.electro_piece = None  # NEW attribute

        self.white_king_location = (7, 4)  # Standard position
        self.black_king_location = (0, 4)  # Standard position
        self.checkmate = False
        self.stalemate = False
        self.in_check = False
        self.pins = []
        self.checks = []
        self.repetition_count = {}
        self.fifty_move_counter = 0
        # En passant
        self.en_passant_possible = (
            None,
            None,
        )  # Coordinates for square where en passant possible
        self.en_passant_possible_log = [self.en_passant_possible]

        # Castling
        self.white_castle_king_side = True
        self.white_castle_queen_side = True
        self.black_castle_king_side = True
        self.black_castle_queen_side = True
        self.castle_rights_log = [
            CastleRights(
                self.white_castle_king_side,
                self.black_castle_king_side,
                self.white_castle_queen_side,
                self.black_castle_queen_side,
            )
        ]
        # Remove snapshot_stack and initialize history using SQL storage
        # self.snapshot_stack = []
        self.history = GameHistorySQL()
        self.redo_stack = []  # NEW: Stack for redo snapshots
        self.history_mode = False  # NEW: Flag to freeze moves in history mode
        self.snapshot_list = []
        self.current_snapshot_index = -1
        # NEW: Take an initial snapshot of the starting state.
        initial_snapshot = self._make_snapshot()
        self.snapshot_list.append(initial_snapshot)
        self.current_snapshot_index = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the SQL history object to avoid pickling sqlite3.Connection.
        if "history" in state:
            del state["history"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize history after unpickling.
        from GameHistorySQL import GameHistorySQL

        self.history = GameHistorySQL()

    def is_checkmate(self):
        """Checks if the current player is in checkmate."""
        if self.in_check:
            valid_moves = self.get_valid_moves()
            if len(valid_moves) == 0:
                return True
        return False

    def is_stalemate(self):
        """Checks if the current player is in stalemate."""
        if not self.in_check:
            valid_moves = self.get_valid_moves()
            if len(valid_moves) == 0:
                return True
        return False

    def _make_snapshot(self):

        state = self.__getstate__()
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(data)
        return compressed

    def _restore_snapshot(self, snapshot):

        data = zlib.decompress(snapshot)
        state = pickle.loads(data)
        if "history" in state:
            del state["history"]
        self.__dict__.update(state)
        self.__post_restore__()

    def make_move(self, move, promotion_choice=None):
        """Executes a move and updates the move log, applying elemental effects."""
        global promoted_piece

        if self.history_mode:
            print("History mode active: moves are disabled.")
            return  # NEW: Prevent making moves while in history mode

        # Before making move, if we’re not at latest snapshot, truncate future snapshots.
        if self.current_snapshot_index < len(self.snapshot_list) - 1:
            self.snapshot_list = self.snapshot_list[: self.current_snapshot_index + 1]

        # Check if the piece is frozen
        if move.piece_moved.frozen_turns > 0:
            print("This piece is frozen and cannot move.")
            return

        # Check if the target piece is shielded
        if move.piece_captured is not None and move.piece_captured.shielded:
            print(
                "Shield broken! The capturing piece returns to its original position."
            )
            move.piece_captured.shielded = False
            self.white_to_move = (
                not self.white_to_move
            )  # Switch turn after breaking shield
            return

        self.board[move.start_row][
            move.start_column
        ] = None  # When a piece is moved, the square it leaves is blank
        self.board[move.end_row][
            move.end_column
        ] = move.piece_moved  # Moves piece to new location
        move.piece_moved.position = (move.end_row, move.end_column)
        self.move_log.append(move)
        if move.piece_moved.piece_type == "K":
            if move.piece_moved.color == "w":
                self.white_king_location = (move.end_row, move.end_column)
            else:
                self.black_king_location = (move.end_row, move.end_column)
        # Pawn promotion (adaptive)
        if move.is_pawn_promotion:
            # Determine the file letter using Move's mapping.
            from ChessEngine import Move  # Ensure Move is imported

            file_letter = Move.columns_to_files.get(move.end_column, "?")
            if file_letter in ["a", "h"]:
                promotion_choice = "R"
            elif file_letter == "e":
                if promotion_choice is None:
                    # Player is allowed to choose when promoting on "e" file.
                    raise ValueError(
                        "Promotion choice must be provided for pawn promotion on the e file."
                    )
            else:
                # For any other file, default to queen unless a choice is provided.
                if promotion_choice is None:
                    promotion_choice = "Q"
            self.board[move.end_row][move.end_column] = {
                "Q": Queen,
                "R": Rook,
                "B": Bishop,
                "N": Knight,
            }[promotion_choice](
                move.piece_moved.color,
                (move.end_row, move.end_column),
                move.piece_moved.identifier,
            )
        # En passant
        if move.is_en_passant_move:
            self.board[move.start_row][move.end_column] = None
        # Before capturing: check if captured piece is shielded
        if move.piece_captured is not None and move.piece_captured.shielded:
            print("Shield broken! Captured shielded piece.")
            move.shield_capture = True
            move.piece_captured.shielded = False
            # Continue with capture instead of aborting the move.
        # If a captured piece is marked to explode, record the flag.
        if move.piece_captured is not None and move.piece_captured.will_explode:
            move.explosive_capture = True
        # Update en passant possibility
        if (
            move.piece_moved.piece_type == "P"
            and abs(move.start_row - move.end_row) == 2
        ):
            self.en_passant_possible = (
                (move.start_row + move.end_row) // 2,
                move.start_column,
            )
        else:
            self.en_passant_possible = (None, None)
        self.en_passant_possible_log.append(self.en_passant_possible)
        # Castling moves - update rook position so it can move later.
        if move.is_castle_move:
            if move.end_column - move.start_column == 2:  # kingside castle
                rook = self.board[move.end_row][move.end_column + 1]
                self.board[move.end_row][move.end_column - 1] = rook
                if rook is not None:
                    rook.position = (move.end_row, move.end_column - 1)
                self.board[move.end_row][move.end_column + 1] = None
            else:  # queenside castle (move.end_column - move.start_column == -2)
                rook = self.board[move.end_row][move.end_column - 2]
                self.board[move.end_row][move.end_column + 1] = rook
                if rook is not None:
                    rook.position = (move.end_row, move.end_column + 1)
                self.board[move.end_row][move.end_column - 2] = None
        # Update castle rights
        self.update_castle_rights(move)
        self.castle_rights_log.append(
            CastleRights(
                self.white_castle_king_side,
                self.black_castle_king_side,
                self.white_castle_queen_side,
                self.black_castle_queen_side,
            )
        )
        # Apply elemental effect if landing on an elemental square.
        if (move.end_row, move.end_column) in self.elementalTiles.tiles:
            element, _ = self.elementalTiles.tiles[(move.end_row, move.end_column)]
            # Use ElementalTiles.apply_effect to trigger the corresponding effect.
            self.elementalTiles.apply_effect(move.piece_moved, element)

        # ...existing move processing code...

        # If a capture occurred on a square with an elemental tile,
        # remove that tile to simulate its consumption.
        if (
            move.piece_captured is not None
            and (move.end_row, move.end_column) in self.elementalTiles.tiles
        ):
            del self.elementalTiles.tiles[(move.end_row, move.end_column)]

        # Update elemental tiles (decrement timers, possibly remove expired ones).
        self.elementalTiles.update(self.board)

        # If a capture occurred and the captured piece is marked to explode, trigger explosion
        if move.piece_captured is not None and move.piece_captured.will_explode:
            self.elementalTiles.explode_piece(self.board, move.end_row, move.end_column)
            while True:
                if (
                    not self.king_is_alive("w")
                    or not self.king_is_alive("b")
                    and self.king_is_alive("w")
                    or not self.king_is_alive("b")
                ):
                    self.checkmate = True
                    break
                elif (
                    not self.king_is_alive("w")
                    and not self.king_is_alive("b")
                    or self.is_draw()
                ):
                    self.stalemate = True
                    break
                else:
                    break
        # Handle electro effect: allow extra move only if the moving piece still has safe moves.
        if move.piece_moved.electro_move:
            extra_moves = move.piece_moved.get_valid_moves(
                self.board, self.pins, self.en_passant_possible, self
            )
            if extra_moves:
                move.is_electro_move = True  # Mark move as electro
                move.piece_moved.electro_move = False  # Clear flag for extra move
                self.electro_piece = move.piece_moved  # Grant extra move
            else:
                self.electro_piece = None
                self.white_to_move = not self.white_to_move
        else:
            if self.electro_piece:
                if len(self.move_log) > 1:
                    self.move_log[-1].is_electro_move = True
                    self.move_log.pop()
            self.electro_piece = None
            self.white_to_move = not self.white_to_move

        # Update repetition count
        board_state = self.get_board_state()
        if board_state in self.repetition_count:
            self.repetition_count[board_state] += 1
        else:
            self.repetition_count[board_state] = 1

        # Update fifty-move counter
        if move.piece_captured is None and move.piece_moved.piece_type != "P":
            self.fifty_move_counter += 1
        else:
            self.fifty_move_counter = 0

        # Check for checkmate or stalemate based on king presence
        self.checkmate = (
            not self.king_is_alive("w")
            or not self.king_is_alive("b")
            or self.is_checkmate()
        )
        self.stalemate = (
            not self.king_is_alive("w")
            and not self.king_is_alive("b")
            or self.is_stalemate()
        )
        # NEW: If a king has been captured, immediately declare checkmate and game over.
        if not self.king_is_alive("w") or not self.king_is_alive("b"):
            print("Game Over: King captured.")
            self.checkmate = True

        # At the end of processing, save new snapshot:
        snapshot = self._make_snapshot()
        self.snapshot_list.append(snapshot)
        self.current_snapshot_index = len(self.snapshot_list) - 1

    def __post_restore__(self):
        # Recalculate dynamic attributes after state restoration.
        self.in_check, self.pins, self.checks = self.check_for_pins_and_checks()
        white_found = False
        black_found = False
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                piece = self.board[r][c]
                if piece is not None and piece.piece_type == "K":
                    if piece.color == "w":
                        self.white_king_location = (r, c)
                        white_found = True
                    else:
                        self.black_king_location = (r, c)
                        black_found = True
        if not white_found:
            self.white_king_location = None
        if not black_found:
            self.black_king_location = None

    def undo_move(self, half_move=False):
        # If undoing half a move, restore the pre-move snapshot;
        # else, pop two snapshots (undo full move).
        if half_move:
            self.history.restore(self)
        else:
            self.history.restore(self)
            self.history.restore(self)
        self.__post_restore__()

    def king_is_alive(self, color):
        for row in self.board:
            for square in row:
                if square and square.piece_type == "K" and square.color == color:
                    return True
        return False

    def get_board_state(self):
        """Returns a string representation of the board state for repetition checking."""
        board_state = ""
        for row in self.board:
            for square in row:
                if square is None:
                    board_state += "--"
                else:
                    board_state += square.color + square.piece_type
        return board_state

    def is_draw(self):
        """Checks for draw conditions: repetition, 50-move rule, and insufficient material."""
        # Check for draw by repetition
        board_state = self.get_board_state()
        if self.repetition_count.get(board_state, 0) >= 3:
            print("Draw by repetition")
            return True

        # Check for draw by 50-move rule
        if self.fifty_move_counter >= 50:
            print("Draw by 50-move rule")
            return True

        # Check for draw by insufficient material
        if self.insufficient_material():
            print("Draw by insufficient material")
            return True

        return False

    def insufficient_material(self):
        """Checks for insufficient material to checkmate."""
        white_pieces = []
        black_pieces = []
        for row in self.board:
            for square in row:
                if square is not None:
                    if square.color == "w":
                        white_pieces.append(square)
                    else:
                        black_pieces.append(square)

        if len(white_pieces) == 1 and len(black_pieces) == 1:
            return True  # King vs King

        if len(white_pieces) == 2 and len(black_pieces) == 1:
            if any(piece.piece_type in ["B", "N"] for piece in white_pieces):
                return True  # King and Bishop/Knight vs King

        if len(white_pieces) == 1 and len(black_pieces) == 2:
            if any(piece.piece_type in ["B", "N"] for piece in black_pieces):
                return True  # King vs King and Bishop/Knight

        return False

    def get_valid_moves(self):
        if self.history_mode:
            return []  # NEW: Disable all moves in history mode
        self.in_check, self.pins, self.checks = self.check_for_pins_and_checks()
        if self.white_to_move:
            king_row, king_column = self.white_king_location
        else:
            king_row, king_column = self.black_king_location
        if self.in_check:
            if len(self.checks) == 1:
                valid_moves = self.get_all_possible_moves()
                check = self.checks[0]
                valid_squares = []
                if self.board[check[0]][check[1]].piece_type == "N":
                    valid_squares = [(check[0], check[1])]
                else:
                    for i in range(1, len(self.board)):
                        sq = (king_row + check[2] * i, king_column + check[3] * i)
                        valid_squares.append(sq)
                        if sq == (check[0], check[1]):
                            break
                valid_moves = [
                    m
                    for m in valid_moves
                    if m.piece_moved
                    and (
                        m.piece_moved.piece_type == "K"
                        or (m.end_row, m.end_column) in valid_squares
                    )
                ]
            else:
                valid_moves = []
                valid_moves.extend(self.get_king_moves(king_row, king_column, []) or [])
        else:
            valid_moves = self.get_all_possible_moves()
        valid_moves = [
            m
            for m in valid_moves
            if m.piece_moved
            and m.piece_moved.frozen_turns == 0
            and self.elementalTiles.is_explosion_safe(
                self.board,
                self.white_king_location,
                self.black_king_location,
                [(m.end_row, m.end_column)],
                self.pins,
                m.piece_moved.color,
            )
        ]
        if len(valid_moves) == 0:
            if self.in_check:
                self.checkmate = True
                self.stalemate = False
            else:
                self.stalemate = True
                self.checkmate = False
        else:
            self.checkmate = False
            self.stalemate = False
        return valid_moves

    def get_all_possible_moves(self):
        """Gets all moves without check filtering by calling each piece's own move generator."""
        moves = []
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                piece = self.board[row][col]
                if piece is not None and piece.color == (
                    "w" if self.white_to_move else "b"
                ):
                    if piece.piece_type == "K":
                        moves.extend(
                            piece.get_valid_moves(
                                self.board, self.pins, self.en_passant_possible, self
                            )
                        )
                    else:
                        moves.extend(
                            piece.get_valid_moves(
                                self.board, self.pins, self.en_passant_possible
                            )
                        )
        return moves

    def add_random_elemental_square(self):
        """Add a random elemental square to the board"""
        self.elementalTiles.add_random_tile(self.board)

    def update_castle_rights(self, move):
        """Updates castling rights based on the move made."""

        # If the king moves, disable castling for that side
        if move.piece_moved.piece_type == "K":
            if move.piece_moved.color == "w":
                self.white_castle_queen_side = False
                self.white_castle_king_side = False
            else:
                self.black_castle_queen_side = False
                self.black_castle_king_side = False

        # If a white rook moves, disable castling on that side
        elif move.piece_moved.piece_type == "R" and move.start_row == 7:
            if move.start_column == 0:
                self.white_castle_queen_side = False
            elif move.start_column == 7:
                self.white_castle_king_side = False

        # If a black rook moves, disable castling on that side
        elif move.piece_moved.piece_type == "R" and move.start_row == 0:
            if move.start_column == 0:
                self.black_castle_queen_side = False
            elif move.start_column == 7:
                self.black_castle_king_side = False

        # If a white rook is captured on its starting square, disable castling
        if (
            move.piece_captured is not None
            and move.piece_captured.piece_type == "R"
            and move.end_row == 7
        ):
            if move.end_column == 0:
                self.white_castle_queen_side = False
            elif move.end_column == 7:
                self.white_castle_king_side = False

        # If a black rook is captured on its starting square, disable castling
        elif (
            move.piece_captured is not None
            and move.piece_captured.piece_type == "R"
            and move.end_row == 0
        ):
            if move.end_column == 0:
                self.black_castle_queen_side = False
            elif move.end_column == 7:
                self.black_castle_king_side = False

    def square_under_attack(self, row, column, ally):
        """Checks outward from a square to see it is being attacked, thus invalidating castling"""
        opponent = "b" if self.white_to_move else "w"
        directions = (
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
        for j in range(len(directions)):
            d = directions[j]
            for i in range(1, len(self.board)):
                end_row = row + d[0] * i
                end_column = column + d[1] * i
                if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):
                    end_piece = self.board[end_row][end_column]
                    if (
                        end_piece is not None and end_piece.color == ally
                    ):  # no attack from that direction
                        break
                    elif end_piece is not None and end_piece.color == opponent:
                        piece_type = end_piece.piece_type
                        if (
                            (0 <= j <= 3 and piece_type == "R")
                            or (4 <= j <= 7 and piece_type == "B")
                            or (
                                i == 1
                                and piece_type == "P"
                                and (
                                    (opponent == "w" and 6 <= j <= 7)
                                    or (opponent == "b" and 4 <= j <= 5)
                                )
                            )
                            or (piece_type == "Q")
                            or (i == 1 and piece_type == "K")
                        ):
                            return True
                        else:  # Enemy piece but not applying check
                            break
                else:  # Off board
                    break

        knight_moves = (
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        )
        for move in knight_moves:
            end_row = row + move[0]
            end_column = column + move[1]
            if 0 <= end_row < len(self.board) and 0 <= end_column < len(self.board):
                end_piece = self.board[end_row][end_column]
                if (
                    end_piece is not None
                    and end_piece.color == opponent
                    and end_piece.piece_type == "N"
                ):
                    return True
        return False

    def check_for_pins_and_checks(self):
        """
        Returns if the player is in check, a list of pins, and a list of checks
        """
        pins = []  # squares pinned and the direction its pinned from
        checks = []  # squares where enemy is applying a check
        in_check = False
        if self.white_to_move:
            enemy_color = "b"
            ally_color = "w"
            start_row = self.white_king_location[0]
            start_col = self.white_king_location[1]
        else:
            enemy_color = "w"
            ally_color = "b"
            start_row = self.black_king_location[0]
            start_col = self.black_king_location[1]

        # Check outward from king for pins and checks
        # Track pins
        directions = (
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
        for j in range(len(directions)):
            d = directions[j]
            possible_pin = ()  # reset possible pins
            for i in range(1, len(self.board)):
                end_row = start_row + d[0] * i
                end_col = start_col + d[1] * i
                if 0 <= end_row < len(self.board) and 0 <= end_col < len(self.board[0]):
                    end_piece = self.board[end_row][end_col]
                    if (
                        end_piece is not None
                        and end_piece.color == ally_color
                        and end_piece.piece_type != "K"
                    ):
                        if possible_pin == ():  # first allied piece could be pinned
                            possible_pin = (end_row, end_col, d[0], d[1])
                        else:  # second allied piece - no pin or check possible in this direction
                            break
                    elif end_piece is not None and end_piece.color == enemy_color:
                        piece_type = end_piece.piece_type
                        # Five possibilities for checks/pins:
                        # 1. Orthogonally away from king and piece is a rook
                        # 2. Diagonally away from king and piece is a bishop
                        # 3. One square away diagonally from king and piece is a pawn
                        # 4. Any direction and piece is a queen
                        # 5. Any direction 1 square away and piece is a king
                        if (
                            (0 <= j <= 3 and piece_type == "R")
                            or (4 <= j <= 7 and piece_type == "B")
                            or (
                                i == 1
                                and piece_type == "P"
                                and (
                                    (enemy_color == "w" and 6 <= j <= 7)
                                    or (enemy_color == "b" and 4 <= j <= 5)
                                )
                            )
                            or (piece_type == "Q")
                            or (i == 1 and piece_type == "K")
                        ):
                            if possible_pin == ():  # no piece blocking, so check
                                in_check = True
                                checks.append((end_row, end_col, d[0], d[1]))
                                break
                            else:  # piece blocking so pin
                                pins.append(possible_pin)
                                break
                        else:  # enemy piece not applying check
                            break
                else:  # off board
                    break

        # Check for knight checks
        knight_moves = (
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        )
        for m in knight_moves:
            end_row = start_row + m[0]
            end_col = start_col + m[1]
            if 0 <= end_row < len(self.board) and 0 <= end_col < len(self.board[0]):
                end_piece = self.board[end_row][end_col]
                if (
                    end_piece is not None
                    and end_piece.color == enemy_color
                    and end_piece.piece_type == "N"
                ):  # enemy knight attacking king
                    in_check = True
                    checks.append((end_row, end_col, m[0], m[1]))

        return in_check, pins, checks

    def get_valid_moves_for_piece(self, piece):
        """Gets all valid moves for a specific piece."""
        moves = []
        self.move_functions[piece.piece_type](
            piece.position[0], piece.position[1], moves
        )
        return moves

    def to_fen(self):
        """Converts the current board state to a FEN string.
        This basic implementation only covers piece placement and active color.
        Castling, en passant, halfmove and fullmove fields are set to default values.
        """
        fen_rows = []
        for row in self.board:
            empty = 0
            fen_row = ""
            for square in row:
                if square is None:
                    empty += 1
                else:
                    if empty:
                        fen_row += str(empty)
                        empty = 0
                    # White pieces are uppercase, black pieces lowercase
                    char = square.piece_type
                    fen_row += char.upper() if square.color == "w" else char.lower()
            if empty:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        # FEN files start from rank 8 (row 0) to rank 1 (row 7)
        placement = "/".join(fen_rows)
        active_color = "w" if self.white_to_move else "b"
        # Use default castling rights, en passant, halfmove and fullmove
        fen = f"{placement} {active_color} KQkq - 0 1"
        return fen

    def navigate_history(self, direction):
        # NEW: Navigate history with "back" (undo) or "forward" (redo)

        if direction == "back":
            if self.current_snapshot_index > 0:
                self.current_snapshot_index -= 1
                self._restore_snapshot(self.snapshot_list[self.current_snapshot_index])
                self.history_mode = True
            else:
                print("Already at the oldest snapshot.")
        elif direction == "forward":
            if self.current_snapshot_index < len(self.snapshot_list) - 1:
                self.current_snapshot_index += 1
                self._restore_snapshot(self.snapshot_list[self.current_snapshot_index])
                self.history_mode = True
            else:
                print("Already at the latest snapshot.")

    def exit_history_mode(self):
        # Restore latest snapshot and exit history mode.
        if self.snapshot_list:
            self.current_snapshot_index = len(self.snapshot_list) - 1
            self._restore_snapshot(self.snapshot_list[self.current_snapshot_index])
        self.history_mode = False

    def get_king_moves(self, row, col, moves):
        # NEW: Return the king's valid moves from its current location.
        king = self.board[row][col]
        if king is not None and king.piece_type == "K":
            return king.get_valid_moves(
                self.board, self.pins, self.en_passant_possible, self
            )
        return []


class Move:
    """
    Class responsible for storing information about particular moves,
    including starting and ending positions, which pieces were moved and captured,
    and special moves such as en passant, pawn promotion, and castling.
    """

    # Update the rank-file mappings for a 8x8 board
    ranks_to_rows = {str(i): 8 - i for i in range(1, 9)}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_columns = {chr(ord("a") + i): i for i in range(8)}
    columns_to_files = {v: k for k, v in files_to_columns.items()}

    def __init__(
        self,
        start_square,
        end_square,
        board,
        en_passant=False,
        pawn_promotion=False,
        castle=False,
        promoted_piece=None,
        is_electro_move=False,
        is_fire_move=False,
    ):
        self.start_row, self.start_column = start_square
        self.end_row, self.end_column = end_square
        self.piece_moved = board[self.start_row][self.start_column]
        self.piece_captured = board[self.end_row][self.end_column]

        # Ensure piece_moved is not None before accessing its attributes
        if self.piece_moved is not None:
            # Pawn promotion check (updated for 8×8 board)
            self.is_pawn_promotion = pawn_promotion or (
                (self.piece_moved.piece_type == "P" and self.end_row == 0)
                or (self.piece_moved.piece_type == "P" and self.end_row == 7)
            )
        else:
            self.is_pawn_promotion = False

        self.promoted_piece = promoted_piece  # Store promoted piece if applicable

        self.is_en_passant_move = en_passant
        if self.is_en_passant_move:
            self.piece_captured = Pawn(
                "w" if self.piece_moved.color == "b" else "b",
                (self.end_row, self.end_column),
                "P",
            )

        self.is_castle_move = castle
        self.is_capture = self.piece_captured is not None
        self.is_electro_move = is_electro_move
        self.is_fire_move = is_fire_move

        # Unique move ID (adjusted for 8×8 board)
        self.move_id = (
            self.start_row * 1000
            + self.start_column * 100
            + self.end_row * 10
            + self.end_column
        )

    def __eq__(self, other):
        """Overrides the equals method for move comparison"""
        return isinstance(other, Move) and self.move_id == other.move_id

    def get_chess_notation(self):
        """Creates a rank and file chess notation"""
        return self.get_rank_file(
            self.start_row, self.start_column
        ) + self.get_rank_file(self.end_row, self.end_column)

    def get_rank_file(self, row, column):
        """Gets chess notation for given rank and file"""
        return self.columns_to_files[column] + self.rows_to_ranks[row]

    def get_disambiguation(self, valid_moves):
        """
        Returns disambiguation characters (file, rank, or both) if needed.
        """
        same_piece_moves = [
            move
            for move in valid_moves
            if move.piece_moved.piece_type == self.piece_moved.piece_type
            and move.end_row == self.end_row
            and move.end_column == self.end_column
        ]

        if len(same_piece_moves) == 1:
            return ""  # No ambiguity

        # Check if other pieces of the same type can move to this square
        same_file = any(
            move.start_column == self.start_column
            for move in same_piece_moves
            if move != self
        )
        same_rank = any(
            move.start_row == self.start_row
            for move in same_piece_moves
            if move != self
        )

        # If both file and rank are the same, use full disambiguation (e.g., Nf3e5)
        if same_file and same_rank:
            return (
                self.columns_to_files[self.start_column]
                + self.rows_to_ranks[self.start_row]
            )

        # If pieces come from the same file, use rank (e.g., N3e5)
        elif same_file:
            return self.rows_to_ranks[self.start_row]

        # If pieces come from the same rank, use file (e.g., Nfe5)
        elif same_rank:
            return self.columns_to_files[self.start_column]

        # Default: Use file unless it's a knight (to avoid confusion with pawn moves)
        else:
            return self.columns_to_files[self.start_column]

    def __str__(self):
        # Get base piece name
        piece = (
            self.piece_moved.piece_type if self.piece_moved.piece_type != "P" else ""
        )

        # Add elemental tags
        if self.piece_moved.shielded:
            piece += "_W"
        if self.piece_moved.frozen_turns > 0:
            piece += "_I"
        if self.is_fire_move:
            piece += "_F"
        if self.is_electro_move:
            piece += "_E"

        # Castling
        if self.is_castle_move:
            move_string = "O-O" if self.end_column > self.start_column else "O-O-O"
        else:
            end_square = self.get_rank_file(self.end_row, self.end_column)
            move_string = ""
            # Pawn moves
            if self.piece_moved.piece_type == "P":
                if self.is_capture and self.is_pawn_promotion:  # Capture and promotion
                    move_string = f"{self.columns_to_files[self.start_column]}x{end_square}={self.promoted_piece}"
                elif self.is_pawn_promotion:  # Promotion without capture
                    move_string = f"{end_square}={self.promoted_piece}"
                elif self.is_capture:  # Capture move
                    move_string = (
                        f"{self.columns_to_files[self.start_column]}x{end_square}"
                    )
                else:  # Normal move
                    move_string = end_square
            else:
                move_string = self.piece_moved.piece_type
                # Instead of calling GameState().get_valid_moves(), skip disambiguation for now.
                disambiguation = ""  # Previously: self.get_disambiguation(GameState().get_valid_moves())
                move_string += disambiguation
                if self.is_capture:
                    move_string += "x"
                move_string += end_square
        # NEW: Prefix extra moves with "e-"
        if self.is_electro_move:
            move_string = "e-" + move_string
        return move_string
