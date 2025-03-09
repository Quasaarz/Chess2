import sqlite3
import pickle
import zlib


class GameHistorySQL:
    def __init__(self, db_path=":memory:"):
        # Use an in-memory SQLite database (or provide a filepath to persist between sessions)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot BLOB)"
        )
        self.conn.commit()

    def add_snapshot(self, game_state):
        # Exclude non-picklable attributes such as 'history' and 'move_functions'
        state = {
            k: v
            for k, v in game_state.__dict__.items()
            if k not in ["history", "move_functions"]
        }
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(data)
        self.cursor.execute("INSERT INTO history (snapshot) VALUES (?)", (compressed,))
        self.conn.commit()

    def pop_snapshot(self):
        self.cursor.execute("SELECT id, snapshot FROM history ORDER BY id DESC LIMIT 1")
        row = self.cursor.fetchone()
        if row:
            id_val, snapshot = row
            self.cursor.execute("DELETE FROM history WHERE id = ?", (id_val,))
            self.conn.commit()
            return snapshot
        return None

    def clear_history(self):
        self.cursor.execute("DELETE FROM history")
        self.conn.commit()

    def restore(self, game_state):
        snapshot = self.pop_snapshot()
        if snapshot is not None:
            data = zlib.decompress(snapshot)
            loaded_state = pickle.loads(data)
            # Do not override 'history'
            if "history" in loaded_state:
                del loaded_state["history"]
            game_state.__dict__.update(loaded_state)
        else:
            print("No moves to undo.")
