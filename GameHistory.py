import pickle
import zlib


class GameHistory:
    def __init__(self):
        self.snapshots = []

    def add_snapshot(self, game_state):
        # Serialize and compress the game state's dictionary for efficiency.
        data = pickle.dumps(game_state.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(data)
        self.snapshots.append(compressed)

    def pop_snapshot(self):
        if self.snapshots:
            return self.snapshots.pop()
        return None

    def clear_history(self):
        self.snapshots = []

    def restore(self, game_state):
        snapshot = self.pop_snapshot()
        if snapshot is not None:
            data = zlib.decompress(snapshot)
            game_state.__dict__ = pickle.loads(data)
        else:
            print("No moves to undo.")
