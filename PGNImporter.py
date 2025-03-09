import os
import pygame as p


def get_game_logs_folder():
    # Returns (and creates if needed) the folder where PGNs are saved.
    logs_dir = os.path.join(os.path.dirname(__file__), "game_logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def list_pgn_files():
    # Lists all PGN files in the game_logs folder.
    logs_dir = get_game_logs_folder()
    files = [f for f in os.listdir(logs_dir) if f.endswith(".pgn")]
    files.sort()
    return files


def load_pgn(file_name):
    # Loads and returns the contents of a given PGN file.
    filepath = os.path.join(get_game_logs_folder(), file_name)
    with open(filepath, "r") as f:
        return f.read()


def replay_pgn(screen, pgn_text):
    """
    Displays the PGN text in a scrollable view.
    Press ESC to exit the replay view.
    """
    clock = p.time.Clock()
    font = p.font.SysFont("Helvetica", 20)
    lines = pgn_text.splitlines()
    scroll = 0
    running = True
    while running:
        for event in p.event.get():
            if event.type == p.QUIT:
                running = False
            elif event.type == p.KEYDOWN:
                if event.key == p.K_ESCAPE:
                    running = False
                elif event.key == p.K_DOWN:
                    scroll = min(scroll + 1, max(0, len(lines) - 10))
                elif event.key == p.K_UP:
                    scroll = max(scroll - 1, 0)
        screen.fill(p.Color("black"))
        for idx, line in enumerate(lines[scroll : scroll + 10]):
            text = font.render(line, True, p.Color("whitesmoke"))
            screen.blit(text, (20, 20 + idx * 25))
        p.display.flip()
        clock.tick(30)


def choose_pgn_file(screen):
    """
    Displays a menu listing available PGN files.
    Use UP/DOWN keys to navigate and ENTER to select; ESC cancels.
    Returns the selected filename or None if cancelled.
    """
    clock = p.time.Clock()
    font = p.font.SysFont("Helvetica", 20)
    files = list_pgn_files()
    if not files:
        prompt = font.render("No PGN files found.", True, p.Color("whitesmoke"))
        screen.fill(p.Color("black"))
        screen.blit(prompt, (20, 20))
        p.display.flip()
        p.time.delay(2000)
        return None
    selected = 0
    running = True
    while running:
        for event in p.event.get():
            if event.type == p.QUIT:
                return None
            elif event.type == p.KEYDOWN:
                if event.key == p.K_DOWN:
                    selected = (selected + 1) % len(files)
                elif event.key == p.K_UP:
                    selected = (selected - 1) % len(files)
                elif event.key == p.K_RETURN:
                    return files[selected]
                elif event.key == p.K_ESCAPE:
                    return None
        screen.fill(p.Color("black"))
        title = font.render(
            "Select a PGN file (UP/DOWN, ENTER to choose)", True, p.Color("whitesmoke")
        )
        screen.blit(title, (20, 20))
        for i, file in enumerate(files):
            color = p.Color("yellow") if i == selected else p.Color("whitesmoke")
            text = font.render(file, True, color)
            screen.blit(text, (40, 60 + i * 25))
        p.display.flip()
        clock.tick(30)
