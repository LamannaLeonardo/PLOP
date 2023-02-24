from collections import defaultdict

class AbstractModel:

    def __init__(self):
        self.states = []
        self.transitions = defaultdict(list)

    def add_transition(self, state_src, action, state_dest):
        self.transitions[state_src.id, action] = state_dest.id

    def add_state(self, state_new):
        self.states.append(state_new)

