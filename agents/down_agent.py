class DownAgent:
    """A simple agent that always chooses the 'down' action."""
    def select_action(self, observation, info=None):
        return 2
