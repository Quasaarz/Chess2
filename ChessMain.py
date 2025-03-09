import pygame as p
import ChessEngine
import ChessAI
from itertools import product
import PGNSaver

# Player settings. Set player_one to True to play as white and player_two to True to play as black.
player_one = True  # Set to False if AI plays as white
player_two = False  # Set to False if AI plays as black

p.init()  # Initialize pygame

board_width = board_height = 680  # Can be adjusted if needed
dimension = 8  # Standard chess board dimensions (8x8)
sq_size = board_height // dimension
label_space = 35  # Increased space for row numbers and column letters
max_fps = 100  # For animations
images = {}
colours = [p.Color("#f7f4f3"), p.Color("#6a7b76")]  # Board colors

# Move log specifications
move_log_panel_width = 300  # Increased width for readability
move_log_panel_height = board_height + label_space  # Adjusted height


def load_images():
    """Initialize a global dictionary of images"""
    pieces = [
        "bR",
        "bN",
        "bB",
        "bQ",
        "bK",
        "bP",
        "wR",
        "wN",
        "wB",
        "wQ",
        "wK",
        "wP",
    ]
    for piece in pieces:
        images[piece] = p.transform.smoothscale(
            p.image.load(f"images/{piece}.png"), (sq_size, sq_size)
        )


def get_player_mode(screen):
    """Display menu to choose game mode:
    1. Player vs Player
    2. Player vs AI - Play as White (AI as Black)
    3. Player vs AI - Play as Black (AI as White)
    Returns a tuple (player_one, player_two) where True indicates the side is controlled by a human.
    """
    font = p.font.SysFont("Helvetica", 36, True)
    prompt1 = font.render("Press 1: PvP", True, p.Color("whitesmoke"))
    prompt2 = font.render("Press 2: PvAI - You as White", True, p.Color("whitesmoke"))
    prompt3 = font.render("Press 3: PvAI - You as Black", True, p.Color("whitesmoke"))
    screen.fill(p.Color("black"))
    screen.blit(
        prompt1, (board_width // 2 - prompt1.get_width() // 2, board_height // 2 - 60)
    )
    screen.blit(
        prompt2, (board_width // 2 - prompt2.get_width() // 2, board_height // 2)
    )
    screen.blit(
        prompt3, (board_width // 2 - prompt3.get_width() // 2, board_height // 2 + 60)
    )
    p.display.flip()
    while True:
        for event in p.event.get():
            if event.type == p.KEYDOWN:
                if event.key == p.K_1:
                    return (True, True)  # Both players human
                elif event.key == p.K_2:
                    return (True, False)  # Human plays White, AI plays Black
                elif event.key == p.K_3:
                    return (False, True)  # AI plays White, Human plays Black


def main():
    """Main function which handles user input and updates graphics"""
    screen = p.display.set_mode(
        (board_width + move_log_panel_width + label_space, board_height + label_space)
    )
    # Get game mode from menu
    player_one, player_two = get_player_mode(screen)
    clock = p.time.Clock()
    screen.fill(p.Color("#0A0903"))
    move_log_font = p.font.SysFont("Arial", 15, False, False)
    game_state = ChessEngine.GameState()
    valid_moves = game_state.get_valid_moves()
    move_made = False
    animate = False
    load_images()
    running = True
    square_selected = ()
    player_clicks = []
    game_over = False
    move_log_scroll = 0
    auto_scroll = False
    # --- Removed snapshots and current_move_index ---
    # snapshots = [copy.deepcopy(game_state)]
    # current_move_index = 0

    # Initialize PGNSaver so that moves are saved continuously.
    pgn_saver = PGNSaver.PGNSaver(white="White", black="Black")

    while running:
        human_turn = (game_state.white_to_move and player_one) or (
            not game_state.white_to_move and player_two
        )
        for event in p.event.get():
            if event.type == p.QUIT:
                running = False
            elif event.type == p.MOUSEBUTTONDOWN:
                if not game_state.history_mode and human_turn and not game_over:
                    location = p.mouse.get_pos()
                    column = (location[0] - label_space) // sq_size
                    row = (location[1] - label_space) // sq_size
                    if 0 <= row < dimension and 0 <= column < dimension:
                        if square_selected == (row, column):
                            square_selected = ()
                            player_clicks = []
                        else:
                            square_selected = (row, column)
                            player_clicks.append(square_selected)
                        if len(player_clicks) == 2:
                            move = ChessEngine.Move(
                                player_clicks[0], player_clicks[1], game_state.board
                            )
                            for valid in valid_moves:
                                if move == valid:
                                    if valid.is_pawn_promotion:
                                        promotion_choice = get_promotion_choice(
                                            screen, move.piece_moved.color
                                        )
                                        game_state.make_move(valid, promotion_choice)
                                    else:
                                        game_state.make_move(valid)
                                    move_made = True
                                    animate = True
                                    square_selected = ()
                                    player_clicks = []
                                    break
                            if not move_made:
                                player_clicks = [square_selected]
                # Handle mouse scroll for move log
                if event.type == p.MOUSEBUTTONDOWN and (
                    event.button == 4 or event.button == 5
                ):
                    if event.button == 4:  # Scroll up
                        move_log_scroll = max(move_log_scroll - 1, 0)
                    elif event.button == 5:  # Scroll down
                        move_log_scroll = min(
                            move_log_scroll + 1, len(game_state.move_log) // 2
                        )
            elif event.type == p.KEYDOWN:
                # NEW: Navigate history using left/right arrows.
                if event.key == p.K_LEFT:
                    game_state.navigate_history("back")
                    valid_moves = game_state.get_valid_moves()
                if event.key == p.K_RIGHT:
                    game_state.navigate_history("forward")
                    valid_moves = game_state.get_valid_moves()
                # NEW: Exit history mode and jump to latest state when "E" is pressed.
                if event.key == p.K_e:
                    game_state.exit_history_mode()
                    valid_moves = game_state.get_valid_moves()
                # Existing key handling for 'z' undo, 'r' reset, UP/DOWN scroll:
                if event.key == p.K_z:  # Undo move when 'z' is pressed
                    game_state.undo_move()
                    move_made = True
                    animate = False
                    game_over = False
                    pgn_saver.update_pgn(game_state.move_log)
                if event.key == p.K_r:  # Reset board when 'r' is pressed
                    game_state = ChessEngine.GameState()
                    valid_moves = game_state.get_valid_moves()
                    square_selected = ()
                    player_clicks = []
                    move_made = False
                    animate = False
                    game_over = False
                    pgn_saver = PGNSaver.PGNSaver(white="White", black="Black")
                if event.key == p.K_UP:  # Scroll up
                    move_log_scroll = max(move_log_scroll - 1, 0)
                    auto_scroll = False
                if event.key == p.K_DOWN:  # Scroll down
                    move_log_scroll = min(
                        move_log_scroll + 1, len(game_state.move_log) // 2
                    )
                    auto_scroll = False

        # Only allow AI moves if not in history mode.
        if not game_state.history_mode and not human_turn and not game_over:
            AI_move = ChessAI.find_best_move(
                game_state, valid_moves
            ) or ChessAI.find_random_move(valid_moves)
            game_state.make_move(AI_move)
            move_made = True
            animate = True

        if move_made:
            if animate:
                animate_move(
                    game_state.move_log[-1], screen, game_state.board, clock, game_state
                )
            valid_moves = game_state.get_valid_moves()
            move_made = False
            animate = False
            if auto_scroll:
                move_log_scroll = len(game_state.move_log) // 2
            pgn_saver.update_pgn(game_state.move_log)
            if (
                not game_state.king_is_alive("w")
                or not game_state.king_is_alive("b")
                or game_state.checkmate
            ):
                draw_endgame_text(screen, "Game Over")
                p.display.flip()
                p.time.delay(3000)
                running = False
                break

        if human_turn and not game_over and not game_state.history_mode:
            add_elemental_squares_periodically(game_state)

        draw_game_state(
            screen,
            game_state,
            square_selected,
            valid_moves,
            move_log_font,
            move_log_scroll,
        )
        # NEW: Overlay a notification if in history mode.
        if game_state.history_mode:
            font = p.font.SysFont("Helvetica", 24, True)
            note = font.render(
                "History Mode: No moves allowed. Press E to exit.", True, p.Color("red")
            )
            screen.blit(note, (20, board_height - 40))
        clock.tick(max_fps)
        p.display.flip()


def draw_board(screen, game_state):
    """Draw squares on the board with proper alignment"""
    screen.fill(p.Color("#0A0903"))  # Ensure the background is black
    font = p.font.SysFont("Arial", 20, True)

    for row, column in product(range(dimension), repeat=2):
        colour = colours[(row + column) % 2]
        p.draw.rect(
            screen,
            colour,
            p.Rect(
                label_space + column * sq_size,
                label_space + row * sq_size,
                sq_size,
                sq_size,
            ),
        )

    # Add column letters (A-H) at the top with black background
    for col in range(dimension):
        label_text = font.render(chr(65 + col), True, p.Color("whitesmoke"))
        label_rect = label_text.get_rect(
            center=(label_space + col * sq_size + sq_size // 2, label_space // 2)
        )
        p.draw.rect(screen, p.Color("#0A0903"), label_rect)  # Black background
        screen.blit(label_text, label_rect)

    # Add row numbers (1-8) with black background
    for row in range(dimension):
        label_text = font.render(str(dimension - row), True, p.Color("whitesmoke"))
        label_rect = label_text.get_rect(
            center=(label_space // 2, label_space + row * sq_size + sq_size // 2)
        )
        p.draw.rect(screen, p.Color("#0A0903"), label_rect)  # Black background
        screen.blit(label_text, label_rect)


def draw_game_state(
    screen, game_state, square_selected, valid_moves, move_log_font, move_log_scroll
):
    """Handles all graphics within the current game state."""
    draw_board(screen, game_state)
    highlight_squares(screen, game_state, square_selected, valid_moves)
    draw_elemental_squares(screen, game_state)
    draw_pieces(screen, game_state.board)
    draw_move_log(screen, game_state, move_log_font, move_log_scroll)


def highlight_squares(screen, game_state, square_selected, valid_moves):
    """Highlights selected square, last move, and valid moves."""
    if square_selected:
        row, column = square_selected
        piece = game_state.board[row][column]
        if piece is not None and piece.color == (
            "w" if game_state.white_to_move else "b"
        ):
            s = p.Surface((sq_size, sq_size))
            s.set_alpha(100)
            s.fill(p.Color("blue"))
            screen.blit(
                s, (label_space + column * sq_size, label_space + row * sq_size)
            )

            # Highlight valid moves for the selected piece
            for move in valid_moves:
                if move.start_row == row and move.start_column == column:
                    s.fill(p.Color("blue"))  # Different color for valid moves
                    screen.blit(
                        s,
                        (
                            label_space + move.end_column * sq_size,
                            label_space + move.end_row * sq_size,
                        ),
                    )

    # Highlights last move
    if game_state.move_log:
        last_move = game_state.move_log[-1]
        s = p.Surface((sq_size, sq_size))
        s.set_alpha(100)
        s.fill(p.Color("blue"))
        screen.blit(
            s,
            (
                label_space + last_move.start_column * sq_size,
                label_space + last_move.start_row * sq_size,
            ),
        )
        screen.blit(
            s,
            (
                label_space + last_move.end_column * sq_size,
                label_space + last_move.end_row * sq_size,
            ),
        )


def draw_pieces(screen, board):
    """Draws pieces on the board with correct alignment"""
    for row in range(dimension):
        for column in range(dimension):
            piece = board[row][column]
            if piece is not None:
                screen.blit(
                    images[f"{piece.color}{piece.piece_type}"],
                    p.Rect(
                        label_space + column * sq_size,
                        label_space + row * sq_size,
                        sq_size,
                        sq_size,
                    ),
                )


def draw_elemental_squares(screen, game_state):
    """Draws elemental squares on the board"""
    # Replace game_state.elemental_squares with game_state.elementalTiles.tiles
    for (row, col), (element, turns) in game_state.elementalTiles.tiles.items():
        s = p.Surface((sq_size, sq_size))
        s.set_alpha(100)
        if element == "W":
            s.fill("green")
        elif element == "I":
            s.fill(p.Color("cyan"))
        elif element == "E":
            s.fill(p.Color("purple"))
        elif element == "F":
            s.fill(p.Color("red"))
        screen.blit(s, (label_space + col * sq_size, label_space + row * sq_size))


def add_elemental_squares_periodically(game_state):
    """Adds random elemental squares periodically"""
    if len(game_state.elementalTiles.tiles) < 5:
        game_state.elementalTiles.add_random_tile(game_state.board)


def draw_move_log(screen, game_state, font, move_log_scroll):
    """Draws the move log with improved alignment and proper chess notation"""
    move_log_area = p.Rect(
        board_width + label_space, 0, move_log_panel_width, move_log_panel_height
    )
    p.draw.rect(screen, p.Color("#0A0903"), move_log_area)
    move_log = game_state.move_log
    move_texts = []

    for i in range(0, len(move_log), 2):
        move_1 = format_move(move_log[i], game_state)
        move_2 = (
            format_move(move_log[i + 1], game_state, is_second_move=True)
            if i + 1 < len(move_log)
            else ""
        )
        move_texts.append(f"{i // 2 + 1}. {move_1} {move_2}")

    padding, text_y = 10, 10  # Adjusted for readability
    for text in move_texts[move_log_scroll:]:
        text_object = font.render(text, True, p.Color("whitesmoke"))
        screen.blit(text_object, move_log_area.move(padding, text_y))
        text_y += text_object.get_height() + 5


def format_move(move, game_state, is_second_move=False):
    """Formats a move string according to standard algebraic chess notation."""
    piece = "" if move.piece_moved.piece_type == "P" else move.piece_moved.piece_type
    destination = move.get_rank_file(
        move.end_row, move.end_column
    )  # Convert to chess notation
    is_capture = move.is_capture
    disambiguation = get_disambiguation(move, game_state)
    # For pawn captures, show starting file.
    if move.piece_moved.piece_type == "P" and is_capture:
        piece = get_rank_file(move.start_row, move.start_column)[0]

    capture_symbol = ""
    if is_capture:
        capture_symbol = "x"
        if hasattr(move, "shield_capture") and move.shield_capture:
            capture_symbol += "$"
        if hasattr(move, "explosive_capture") and move.explosive_capture:
            capture_symbol += "?"

    move_string = piece + disambiguation + capture_symbol + destination

    # Pawn promotion
    if move.is_pawn_promotion:
        move_string += f"={move.promoted_piece or 'Q'}"

    # Prefix the second move with "e-" if it is an electro move
    if is_second_move and move.is_electro_move:
        move_string = "e-" + move_string

    # Append check (+) or checkmate (#) indicators if this is the last move in the log.
    if game_state.move_log and move == game_state.move_log[-1]:
        if game_state.checkmate:
            move_string += "#"
        elif game_state.in_check:
            move_string += "+"

    return move_string


def get_disambiguation(move, game_state):
    """Returns file/rank disambiguation if multiple pieces of the same type can move to the same square."""
    similar_pieces = [
        m
        for m in game_state.get_valid_moves()
        if m.piece_moved.piece_type == move.piece_moved.piece_type
        and m.end_row == move.end_row
        and m.end_column == move.end_column
        and m != move
    ]
    if not similar_pieces:
        return ""

    same_rank = any(m.start_row == move.start_row for m in similar_pieces)
    same_file = any(m.start_column == move.start_column for m in similar_pieces)

    if same_file and same_rank:
        return get_rank_file(
            move.start_row, move.start_column
        )  # Use full square notation (e.g., Nbd2)
    elif same_file:
        return get_rank_file(move.start_row, move.start_column)[
            1
        ]  # Use rank (e.g., N3d2)
    elif same_rank:
        return get_rank_file(move.start_row, move.start_column)[
            0
        ]  # Use file (e.g., Nbd2)
    return ""  # No disambiguation needed


def get_rank_file(row, col):
    """Converts board matrix coordinates to standard chess notation (e.g., row 7, col 4 → 'e2')."""
    files = "abcdefgh"  # Columns are files
    ranks = "87654321"  # Rows are ranks
    return files[col] + ranks[row]  # Convert indices to notation


def animate_move(move, screen, board, clock, game_state):
    """Animates a move with proper alignment for the current screen"""
    delta_row = move.end_row - move.start_row  # Change in row
    delta_column = move.end_column - move.start_column  # Change in column
    frames_per_square = 2  # Controls animation speed (frames to move one square)
    frame_count = (abs(delta_row) + abs(delta_column)) * frames_per_square

    for frame in range(frame_count + 1):  # Need +1 to complete the entire animation

        # Frame/frame_count indicates how far along the action is
        row = move.start_row + delta_row * frame / frame_count
        column = move.start_column + delta_column * frame / frame_count

        # Adjust coordinates for alignment
        x = label_space + column * sq_size
        y = label_space + row * sq_size

        # Draw board and pieces for each frame of the animation
        draw_board(screen, game_state)
        draw_pieces(screen, board)
        draw_elemental_squares(screen, game_state)

        # Erase the piece from its ending square
        colour = colours[(move.end_row + move.end_column) % 2]
        end_square = p.Rect(
            label_space + move.end_column * sq_size,
            label_space + move.end_row * sq_size,
            sq_size,
            sq_size,
        )
        p.draw.rect(screen, colour, end_square)

        # Draws a captured piece onto the rectangle if a piece is captured
        if move.piece_captured is not None:
            if move.is_en_passant_move:
                en_passant_row = (
                    move.end_row + 1
                    if move.piece_captured.color == "b"
                    else move.end_row - 1
                )
                end_square = p.Rect(
                    label_space + move.end_column * sq_size,
                    label_space + en_passant_row * sq_size,
                    sq_size,
                    sq_size,
                )
            screen.blit(
                images[f"{move.piece_captured.color}{move.piece_captured.piece_type}"],
                end_square,
            )

        # Draw moving piece at its new animated position
        screen.blit(
            images[f"{move.piece_moved.color}{move.piece_moved.piece_type}"],
            p.Rect(x, y, sq_size, sq_size),
        )

        p.display.flip()
        clock.tick(60)  # Controls frame rate per second for the animation


def draw_promotion_popup(screen, color):
    """Draws a popup for pawn promotion"""
    popup_width, popup_height = 200, 200
    popup_x = (board_width + move_log_panel_width + label_space - popup_width) // 2
    popup_y = (board_height + label_space - popup_height) // 2
    p.draw.rect(screen, p.Color("gray"), (popup_x, popup_y, popup_width, popup_height))
    p.draw.rect(
        screen, p.Color("black"), (popup_x, popup_y, popup_width, popup_height), 2
    )

    pieces = ["Q", "R", "B", "N"]
    piece_images = [images[f"{color}{piece}"] for piece in pieces]
    for i, piece_image in enumerate(piece_images):
        piece_x = popup_x + (i % 2) * (popup_width // 2)
        piece_y = popup_y + (i // 2) * (popup_height // 2)
        screen.blit(piece_image, (piece_x, piece_y))
        p.draw.rect(
            screen,
            p.Color("black"),
            (piece_x, piece_y, popup_width // 2, popup_height // 2),
            2,
        )

    p.display.flip()

    return popup_x, popup_y, popup_width, popup_height


def get_promotion_choice(screen, color):
    """Gets the user's choice for pawn promotion"""
    popup_x, popup_y, popup_width, popup_height = draw_promotion_popup(screen, color)
    while True:
        for event in p.event.get():
            if event.type == p.MOUSEBUTTONDOWN:
                x, y = event.pos
                if (
                    popup_x <= x < popup_x + popup_width
                    and popup_y <= y < popup_y + popup_height
                ):
                    col = (x - popup_x) // (popup_width // 2)
                    row = (y - popup_y) // (popup_height // 2)
                    return ["Q", "R", "B", "N"][row * 2 + col]


def draw_endgame_text(screen, text):
    font = p.font.SysFont("Helvetica", 32, True, False)
    text_object = font.render(text, True, p.Color("gray"), p.Color("mintcream"))
    text_location = p.Rect(0, 0, board_width, board_height).move(
        board_width / 2 - text_object.get_width() / 2,
        board_height / 2 - text_object.get_height() / 2,
    )
    screen.blit(text_object, text_location)

    # Creates a shadowing effect
    text_object = font.render(text, True, p.Color("black"))
    screen.blit(text_object, text_location.move(2, 2))


if __name__ == "__main__":
    main()
