
class State:

    def __init__(self, state_id, perceptions, visible_objects, visible_predicates, gt_state):
        self.id = state_id
        self.perceptions = perceptions
        self.visible_objects = visible_objects
        self.visible_predicates = visible_predicates
        self.self_supervised_predicates = []
        self.gt_state = {'objects': gt_state.metadata['objects'], 'instance_detections2D': gt_state.instance_detections2D}
