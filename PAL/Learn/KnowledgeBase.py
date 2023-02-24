import copy
from collections import defaultdict

import Configuration
from PAL.Learn.EnvironmentModels.AbstractModel import AbstractModel
from PAL.Learn.EnvironmentModels.MapModel import MapModel
from Utils import PddlParser, Logger
import numpy as np


class KnowledgeManager:

    def __init__(self):
        self.all_objects = defaultdict(list)
        self.all_predicates = []
        self.self_supervised_predicates = []
        self.supervised_objects = defaultdict(list)

        self.objects_counting = defaultdict(int)
        self.objects_avg_score = defaultdict(float)

        self.current_state = None

        # Set map model
        self.map_model = MapModel()

        # Abstract model (Finite state machine)
        self.fsm_model = AbstractModel()

        self.unexisting_objects = defaultdict(list)


    def update_objects(self, new_objects, agent_pos):
        position_threshold = 0.2  # if some x y z coordinate is above threshold, then the object instance is a new one

        # List of objects merged with the current knowledge ones, i.e., if a visible object is already existing in the
        # current knowledge, its ID is updated with the already existing one.
        merged_objects = {obj_type: [] for obj_type in new_objects.keys()}

        # Add new objects to current objects dictionary
        for obj_type in new_objects.keys():

            if obj_type.lower() in Configuration.BIG_OBJECTS:
                position_threshold = 1.5
            elif obj_type.lower() in Configuration.MEDIUM_OBJECTS:
                position_threshold = 0.4

            for new_obj_type_inst in new_objects[obj_type]:

                new_obj_x, new_obj_y, new_obj_z = new_obj_type_inst['map_x'], new_obj_type_inst['map_y'], new_obj_type_inst['map_z']

                new_obj_exists = False
                merged_object = None

                # Check if an object instance does not exist yet in the current objects dictionary
                same_type_distances = [np.linalg.norm(np.array([existing_obj['map_x'], existing_obj['map_y'], existing_obj['map_z']])
                                      - np.array([new_obj_x, new_obj_y, new_obj_z]))
                                       for existing_obj in self.all_objects[obj_type]]
                min_distance = min(same_type_distances, default=np.inf)
                if min_distance < position_threshold:
                    existing_obj = self.all_objects[obj_type][np.argmin(same_type_distances)]

                    # Update existing object average score
                    self.objects_avg_score[existing_obj['id']] = new_obj_type_inst['score']/(self.objects_counting[existing_obj['id']] + 1) \
                                            + (self.objects_counting[existing_obj['id']]*self.objects_avg_score[existing_obj['id']])/(self.objects_counting[existing_obj['id']] + 1)

                    # Update objects counting
                    self.objects_counting[existing_obj['id']] += 1

                    merged_object = copy.deepcopy(new_obj_type_inst)
                    merged_object['id'] = existing_obj['id']

                    # Update existing object bounding box
                    existing_obj['bb'] = merged_object['bb']

                    # Update existing object position
                    existing_obj['map_x'] = merged_object['map_x']/(self.objects_counting[existing_obj['id']] + 1) \
                                            + (self.objects_counting[existing_obj['id']]*existing_obj['map_x'])/(self.objects_counting[existing_obj['id']] + 1)
                    existing_obj['map_y'] = merged_object['map_y']/(self.objects_counting[existing_obj['id']] + 1) \
                                            + (self.objects_counting[existing_obj['id']]*existing_obj['map_y'])/(self.objects_counting[existing_obj['id']] + 1)
                    existing_obj['map_z'] = merged_object['map_z']/(self.objects_counting[existing_obj['id']] + 1) \
                                            + (self.objects_counting[existing_obj['id']]*existing_obj['map_z'])/(self.objects_counting[existing_obj['id']] + 1)

                    new_obj_exists = True

                # If new object instance does not exist yet in the current objects dictionary
                if not new_obj_exists:
                    # Update new object id
                    new_obj_count = max([int(obj['id'].split('_')[1]) + 1
                                         for obj in self.all_objects[obj_type]], default=0)
                    new_obj_id = f"{obj_type}_{new_obj_count}"

                    # Object instance is a new one
                    new_obj_type_inst['id'] = new_obj_id

                    # Update objects counting
                    self.objects_counting[new_obj_id] += 1
                    self.objects_avg_score[new_obj_id] = new_obj_type_inst['score']

                    self.all_objects[obj_type].append(new_obj_type_inst)
                    merged_object = copy.deepcopy(new_obj_type_inst)

                # Add a new object to merged ones
                if merged_object is not None:
                    existing_obj_ids = [obj['id'] for obj in merged_objects[obj_type]]
                    if merged_object['id'] not in existing_obj_ids:
                        merged_objects[obj_type].append(merged_object)

        self.update_all_obj_distances(agent_pos)

        return merged_objects


    def add_predicate(self, new_predicate):
        if new_predicate not in self.all_predicates:
            self.all_predicates.append(new_predicate)


    def remove_predicate(self, removed_predicate):
        self.all_predicates = [p for p in self.all_predicates if p != removed_predicate]


    def update_all_predicates(self, new_predicates):

        self.all_predicates = list(set(self.all_predicates + new_predicates))

        # Check "hand_free" predicate
        hand_free = len([o for o in self.all_predicates if o.strip().startswith("holding")]) == 0
        if hand_free and "hand_free()" not in self.all_predicates:
            self.all_predicates.append("hand_free()")

        # Update "viewing(object)" predicate for all visible objects
        self.all_predicates = [pred for pred in self.all_predicates if not pred.startswith("viewing")]
        [self.all_predicates.append(p) for p in new_predicates if p.startswith("viewing")]

        removed_close_to = []
        for pred in [p for p in self.all_predicates if p not in new_predicates]:
            if pred.startswith('close_to'):
                obj_id = pred.split('(')[1].lower().strip()[:-1]
                obj_type = obj_id.split('_')[0]

                obj = [obj for obj in self.all_objects[obj_type] if obj['id'] == obj_id][0]

                if obj['distance'] > Configuration.CLOSE_TO_OBJ_DISTANCE:
                    removed_close_to.append(pred)

        self.all_predicates = [pred for pred in self.all_predicates if pred not in removed_close_to]


        [self.all_predicates.append("close_to({})".format(obj['id']))
         for obj_type in list(self.all_objects.keys())
         for obj in self.all_objects[obj_type]
         if obj['distance'] <= Configuration.CLOSE_TO_OBJ_DISTANCE]


    def update_pddl_state(self):
        PddlParser.update_pddl_state(self.all_objects, self.all_predicates,
                                     self.objects_counting, self.objects_avg_score)


    def update_obj_position(self, obj_id, pos):
        obj_type = obj_id.split("_")[0]
        obj = [obj for obj in self.all_objects[obj_type] if obj['id'] == obj_id][0]

        # Update object coordinates
        obj['map_x'] = pos['x']
        obj['map_y'] = pos['y']
        obj['map_z'] = pos['z']


    def update_all_obj_position(self, micro_action_name, micro_action_result, macro_action_name):

        # If action has failed then no updates are done
        if not micro_action_result.metadata['lastActionSuccess']:
            return

        # Get updated (after executing action) agent hand position
        # AI2THOR 3.3.1
        # hand_pos = {'x':micro_action_result.metadata['hand']['position']['x'],
        #             'y':micro_action_result.metadata['hand']['position']['z'],
        #             'z':micro_action_result.metadata['hand']['position']['y']}

        # AI2THOR >= 3.5
        hand_pos = {'x':micro_action_result.metadata['heldObjectPose']['position']['x'],
                    'y':micro_action_result.metadata['heldObjectPose']['position']['z'],
                    'z':micro_action_result.metadata['heldObjectPose']['position']['y']}

        # Update held object coordinates
        if micro_action_name.startswith("Move") or micro_action_name.startswith("Rotate") \
                or micro_action_name.startswith("Home") or micro_action_name.startswith("Look"):
            held_obj_id = [pred.split("(")[1][:-1].strip() for pred in self.all_predicates if pred.startswith("holding(")]
            if len(held_obj_id) > 0:
                held_obj_id = held_obj_id[0]
                self.update_obj_position(held_obj_id, hand_pos)

                # Update coordinates of all objects contained into held one
                contained_objs = [pred.split("(")[1][:-1].strip().split(",")[0] for pred in self.all_predicates
                                  if pred.startswith("on(") and pred.split("(")[1][:-1].strip().split(",")[1] == held_obj_id]
                for contained_obj_id in contained_objs:
                    self.update_obj_position(contained_obj_id, hand_pos)

        # Update picked object coordinates
        elif micro_action_name.startswith("Pickup"):

            obj_id = macro_action_name.split("(")[1][:-1].lower().strip()
            self.update_obj_position(obj_id, hand_pos)

            # Update coordinates of all objects contained into picked one
            contained_objs = [pred.split("(")[1][:-1].strip().split(",")[0] for pred in self.all_predicates
                              if pred.startswith("on(") and pred.split("(")[1][:-1].strip().split(",")[1] == obj_id]
            for contained_obj_id in contained_objs:
                self.update_obj_position(contained_obj_id, hand_pos)

        # Update put down object coordinates
        elif micro_action_name.startswith("PutObject"):

            contained_obj_id = macro_action_name.split("(")[1][:-1].lower().strip().split(",")[0].strip()
            container_obj_id = macro_action_name.split("(")[1][:-1].lower().strip().split(",")[1].strip()
            container_obj_type = container_obj_id.split("_")[0]
            container_obj = [obj for obj in self.all_objects[container_obj_type] if obj['id'] == container_obj_id][0]
            container_pos = {'x': container_obj['map_x'], "y": container_obj['map_y']}
            self.update_obj_position(contained_obj_id, container_pos)

            # Update coordinates of all objects contained into picked one
            contained_objs = [pred.split("(")[1][:-1].strip().split(",")[0] for pred in self.all_predicates
                              if pred.startswith("on(") and pred.split("(")[1][:-1].strip().split(",")[1] == contained_obj_id]
            for contained_obj_id in contained_objs:
                self.update_obj_position(contained_obj_id, container_pos)


    def update_all_obj_distances(self, agent_pos):

        for obj_type, obj_instances in self.all_objects.items():
            for obj in obj_instances:
                obj_pos = [obj['map_x'], obj['map_y'], obj['map_z']]
                obj['distance'] = np.linalg.norm(np.array(agent_pos) - np.array(obj_pos))


    def remove_object(self, removed_obj_id):

        Logger.write(f'INFO: Removing object {removed_obj_id}')

        # Remove object from all objects list
        removed_obj_type = removed_obj_id.split('_')[0]
        removed_obj = [obj for obj in self.all_objects[removed_obj_type]
                       if obj['id'] == removed_obj_id][0]

        assert len([obj for obj in self.all_objects[removed_obj_type] if obj['id'] == removed_obj_id]) == 1

        self.all_objects[removed_obj_type] = [obj for obj in self.all_objects[removed_obj_type]
                                              if not obj['id'] == removed_obj_id]

        # Reset counter of object id instances
        self.objects_counting[removed_obj_id] = 0
        self.objects_avg_score[removed_obj_id] = 0

        # Remove from the states all self supervised predicates involving the removed object
        for s in self.fsm_model.states:
            s.self_supervised_predicates = [el for el in s.self_supervised_predicates
                                            if removed_obj_id not in el.split('(')[1][:-1].strip().split(',')]
            # Add objects to unexisting ones
            self.unexisting_objects[removed_obj_type].append([removed_obj['map_x'], removed_obj['map_y'], removed_obj['map_z']])

            s.visible_predicates = [el for el in s.visible_predicates
                                    if removed_obj_id not in el.split('(')[1][:-1].strip().split(',')]

        for state in self.fsm_model.states:
            if removed_obj_type in state.visible_objects.keys():
                state.visible_objects[removed_obj_type] = [obj for obj in state.visible_objects[removed_obj_type]
                                                           if obj['id'] != removed_obj_id]

        # Remove all predicates involving the removed object
        self.all_predicates = [el for el in self.all_predicates
                               if removed_obj_id not in el.split('(')[1][:-1].strip().split(',')]

        # Remove from the knowledge base all self supervised predicates involving the removed object
        self.self_supervised_predicates = [el for el in self.self_supervised_predicates
                                           if removed_obj_id not in el.split('(')[1][:-1].strip().split(',')]

        self.update_pddl_state()
