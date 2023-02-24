import copy
from collections import defaultdict

import Configuration
from PAL.Learn.Mapper import Mapper
from PAL.Learn.ObjectDetector import ObjectDetector
from PAL.Learn.SceneClassifier import SceneClassifier
import numpy as np

from Utils import Logger, PddlParser
from Utils.depth_util import get_mindist_xyz_from_depth


class Learner:

    def __init__(self, knowledge_base):

        # Set knowledge base
        self.knowledge_base = knowledge_base

        # Depth mapper
        self.mapper = Mapper(self.knowledge_base)

        # Object detector
        self.object_detector = ObjectDetector()

        # Scene classifier
        self.scene_classifier = SceneClassifier()

        # Memory of agent views when opening e successively closing an object
        self.open_view_memory = None

        self.unexisting_objects = defaultdict(list)


    def add_current_state(self):
        self.knowledge_base.fsm_model.add_state(self.knowledge_base.current_state)


    def add_state(self, state_new):
        self.knowledge_base.fsm_model.add_state(state_new)


    def add_transition(self, state_src, action, state_dest):
        self.knowledge_base.fsm_model.add_transition(state_src, action, state_dest)


    def update_topview(self, file_name, depth_matrix, collision=False):
        angle = self.knowledge_base.current_state.perceptions['agent_angle']
        cam_angle = self.knowledge_base.current_state.perceptions['camera_tilt']
        pos = self.knowledge_base.current_state.perceptions['position']
        self.mapper.update_topview(depth_matrix, file_name, angle, cam_angle, pos, collision)  # depth_matrix in meters


    def get_visible_objects(self, rgb_img, depth_img, agent_pos, agent_angle, event):

        if not Configuration.GROUND_TRUTH_OBJS:
            pred_objects = self.object_detector.get_visible_objects(rgb_img)
        else:
            pred_objects = self.object_detector.get_visible_objects_ground_truth(event)

        pred_objects['labels'] = [obj_type.lower() for obj_type in pred_objects['labels']]

        visible_objects = defaultdict(list)

        for obj_type, obj_bb, obj_score in zip(pred_objects['labels'], pred_objects['boxes'], pred_objects['scores']):
            # Get object centroid from bbox = [x0, y0, x1, y1]. Notice that y0 and y1 are from top to bottom
            obj_centroid = [int(round((obj_bb[2] + obj_bb[0]) / 2)),  # columns (x-axis)
                            int(round((obj_bb[3] + obj_bb[1]) / 2))]  # rows (y-axis)

            # Get object distance by averaging object reduced bbox depth values
            obj_bb_size = [obj_bb[2] - obj_bb[0], obj_bb[3] - obj_bb[1]]  # [height, width]

            # Filter non object bbox values
            depth_matrix = copy.deepcopy(depth_img)

            # Remove first rows
            min_row = max(0, int(round(obj_centroid[1]) - (obj_bb_size[1] * 0.25)) - 1)
            depth_matrix[:min_row, :] = np.nan
            # Remove first cols
            min_col = max(0, int(round(obj_centroid[0]) - (obj_bb_size[0] * 0.25)) - 1)
            depth_matrix[:, :min_col] = np.nan
            # Remove last rows
            max_row = max(0, int(round(obj_centroid[1]) + (obj_bb_size[1] * 0.25)) + 1)
            depth_matrix[max_row:, :] = np.nan
            # Remove last cols
            max_col = max(0, int(round(obj_centroid[0]) + (obj_bb_size[0] * 0.25)) + 1)
            depth_matrix[:, max_col:] = np.nan

            cam_angle = -int(event.metadata['agent']['cameraHorizon'])
            x_obj, y_obj, z_obj = get_mindist_xyz_from_depth(depth_matrix, agent_angle, cam_angle, agent_pos)

            obj_distance = np.linalg.norm(agent_pos - np.array([x_obj, y_obj, z_obj]))

            if x_obj * 100 <= Configuration.MAP_X_MIN or x_obj * 100 >= Configuration.MAP_X_MAX \
                or y_obj * 100 <= Configuration.MAP_Y_MIN or y_obj * 100 >= Configuration.MAP_Y_MAX:
                if obj_type != 'window':
                    Logger.write(f"WARNING: Removing an object of type {obj_type} from the predicted one since its position "
                                 f"is outside of map size.")


            unexisting_obj = False
            new_obj_pos = [x_obj, y_obj, z_obj]

            # Check if object position is close to an unexisting object one
            for obj_pos in self.knowledge_base.unexisting_objects[obj_type]:
                if np.linalg.norm(np.array(obj_pos) - np.array(new_obj_pos)) < 0.2:
                    # DEBUG
                    Logger.write(f"WARNING: Removing an object of type {obj_type} from the predicted one since its "
                                 f"position is too close to an unexisting object of the same type.")
                    unexisting_obj = True
                    break

            # Check if object is well visible, otherwise do not consider it
            view_margin_percentage = 0.1
            x_center = obj_centroid[0]
            y_center = obj_centroid[1]

            bbox_well_visible = obj_bb[0] > 1 \
                               and obj_bb[2] < Configuration.FRAME_WIDTH - 1 \
                               and obj_bb[1] > 1 \
                               and obj_bb[3] < Configuration.FRAME_HEIGHT - 1
            if not bbox_well_visible:
                obj_well_visible = int(Configuration.FRAME_WIDTH * view_margin_percentage) <= x_center <= \
                                   int(Configuration.FRAME_WIDTH * (1 - view_margin_percentage)) \
                                   and int(Configuration.FRAME_HEIGHT * view_margin_percentage) <= y_center <= \
                                   int(Configuration.FRAME_HEIGHT * (1 - view_margin_percentage))
            else:
                obj_well_visible = True

            if not obj_well_visible:
                # DEBUG
                # Logger.write(f"WARNING: Removing an object of type {obj_type} from the predicted one since its bbox "
                #              f"center is too close to the egocentric view edges, i.e., it is likely that the object "
                #              f"is only partially visible.")
                unexisting_obj = True


            if not unexisting_obj:
                visible_objects[obj_type].append({'id': '{}_{}'.format(obj_type, len(visible_objects[obj_type])),
                                                  'map_x': x_obj,
                                                  'map_y': y_obj,
                                                  'map_z': z_obj,
                                                  'distance': obj_distance,
                                                  'bb': {'center': obj_centroid,
                                                         'corner_points': list(obj_bb),
                                                         'size': [obj_bb[2] - obj_bb[0], obj_bb[3] - obj_bb[1]]},
                                                  'score': obj_score
                                                  })
        return visible_objects


    def update_opened_object(self, opened_object_id, depth_img):

        opened_obj_type = opened_object_id.split('_')[0]
        opened_obj = [o for o in self.knowledge_base.all_objects[opened_obj_type] if o['id'] == opened_object_id][0]

        # Get object centroid from bbox = [x0, y0, x1, y1]. Notice that y0 and y1 are from top to bottom
        obj_centroid = opened_obj['bb']['center']

        # Get object distance by averaging object reduced bbox depth values
        obj_bb_size = opened_obj['bb']['size']

        # Filter non object bbox values
        depth_matrix = copy.deepcopy(depth_img)

        # Remove first rows
        min_row = max(0, int(round(obj_centroid[1]) - (obj_bb_size[1] * 0.25)) - 1)
        depth_matrix[:min_row, :] = np.nan
        # Remove first cols
        min_col = max(0, int(round(obj_centroid[0]) - (obj_bb_size[0] * 0.25)) - 1)
        depth_matrix[:, :min_col] = np.nan
        # Remove last rows
        max_row = max(0, int(round(obj_centroid[1]) + (obj_bb_size[1] * 0.25)) + 1)
        depth_matrix[max_row:, :] = np.nan
        # Remove last cols
        max_col = max(0, int(round(obj_centroid[0]) + (obj_bb_size[0] * 0.25)) + 1)
        depth_matrix[:, max_col:] = np.nan

        # Estimate new position according to previous bounding box (i.e. the closed object one)
        agent_pos = self.knowledge_base.current_state.perceptions['position']
        agent_angle = self.knowledge_base.current_state.perceptions['agent_angle']
        cam_angle = self.knowledge_base.current_state.perceptions['camera_tilt']
        x_obj, y_obj, z_obj = get_mindist_xyz_from_depth(depth_matrix, agent_angle, cam_angle, agent_pos)

        opened_obj_distance = np.linalg.norm(agent_pos - np.array([x_obj, y_obj, z_obj]))

        opened_obj['map_x'] = x_obj
        opened_obj['map_y'] = y_obj
        opened_obj['map_z'] = z_obj
        opened_obj['distance'] = opened_obj_distance


    def update_supervised_objects(self, new_state):

        state_objs_id = [o['id'] for v in new_state.visible_objects.values() for o in v]

        for p in [p for p in new_state.self_supervised_predicates
                  if p.startswith('open') or p.startswith('not_open') or p.startswith('toggled') or p.startswith('not_toggled')
                     or p.startswith('filled') or p.startswith('not_filled') or p.startswith('dirty') or p.startswith('not_dirty')]:

            p_objs = [o.strip() for o in p.split('(')[1].strip()[:-1].split(',')]

            for p_obj in p_objs:

                if p_obj in state_objs_id:
                    p_obj_type = p_obj.split('_')[0]

                    obj_inst = [o for o in new_state.visible_objects[p_obj_type] if o['id'] == p_obj][0]
                    obj_bbox = obj_inst['bb']['corner_points']
                    bbox_margin = 15
                    state_rgb = new_state.perceptions['rgb']
                    example_features = state_rgb[
                                       max(0, obj_bbox[1] - bbox_margin): min(state_rgb.shape[0], obj_bbox[3] + (bbox_margin + 1)),
                                       max(0, obj_bbox[0] - bbox_margin): min(state_rgb.shape[1], obj_bbox[2] + (bbox_margin + 1)),
                                       :
                                       ]
                    self.knowledge_base.supervised_objects[p_obj_type].append({'example': example_features,
                                                                               'label': p})

                    supervised_open = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                           if 'open' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS
                    supervised_closed = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                             if 'not_open' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS

                    supervised_toggled = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                           if 'toggled' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS
                    supervised_nottoggled = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                             if 'not_toggled' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS

                    supervised_filled = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                             if 'filled' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS
                    supervised_notfilled = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                             if 'not_filled' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS

                    supervised_dirty = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                             if 'dirty' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS
                    supervised_notdirty = len([p for p in self.knowledge_base.supervised_objects[p_obj_type]
                                             if 'not_dirty' in p['label']]) >= Configuration.MAX_OBJ_SUPERVISIONS

                    if supervised_open:
                        if f'supervised_opened_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_opened_{p_obj_type}()')
                            current_goal = PddlParser.get_goal()
                            assert current_goal is not None
                            new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (open ?o1)', f'(forall (?o1 - {p_obj_type}) (and (not (open ?o1))')
                            PddlParser.set_goal(new_goal)

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                    if supervised_closed:
                        if f'supervised_notopened_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_notopened_{p_obj_type}()')

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                    if supervised_toggled:
                        if f'supervised_toggled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_toggled_{p_obj_type}()')
                            current_goal = PddlParser.get_goal()
                            assert current_goal is not None
                            new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (toggled ?o1)', f'(forall (?o1 - {p_obj_type}) (and (not (toggled ?o1))')
                            PddlParser.set_goal(new_goal)

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                    if supervised_nottoggled:
                        if f'supervised_nottoggled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_nottoggled_{p_obj_type}()')

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                    if supervised_filled:
                        if f'supervised_filled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_filled_{p_obj_type}()')
                            current_goal = PddlParser.get_goal()
                            assert current_goal is not None
                            new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (filled ?o1)', f'(forall (?o1 - {p_obj_type}) (and (not (filled ?o1))')
                            PddlParser.set_goal(new_goal)

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                    if supervised_notfilled:
                        if f'supervised_notfilled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_notfilled_{p_obj_type}()')

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                    if supervised_dirty:
                        if f'supervised_dirty_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_dirty_{p_obj_type}()')
                            current_goal = PddlParser.get_goal()
                            assert current_goal is not None
                            new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (dirty ?o1)', f'(forall (?o1 - {p_obj_type}) (and (not (dirty ?o1))')
                            PddlParser.set_goal(new_goal)

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                    if supervised_notdirty:
                        if f'supervised_notdirty_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                            self.knowledge_base.add_predicate(f'supervised_notdirty_{p_obj_type}()')

                            self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                                  if not f'scanned({p_obj_type}' in p
                                                                  and not f'manipulated({p_obj_type}' in p]

                            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                                              if not f'scanned({p_obj_type}' in p
                                                                              and not f'manipulated({p_obj_type}' in p]
                else:
                    Logger.write(f'WARNING: discarding supervision of object {p_obj} since not recognized in state {new_state.id}')


        for p in [p for p in self.knowledge_base.self_supervised_predicates if p.startswith('scanned')]:

            p_obj = [o.strip() for o in p.split('(')[1].strip()[:-1].split(',')][0]
            p_obj_type = p_obj.split('_')[0]
            supervised_open = self.check_supervised_open(p_obj_type)
            supervised_closed = self.check_supervised_close(p_obj_type)
            supervised_toggled = self.check_supervised_toggled(p_obj_type)
            supervised_nottoggled = self.check_supervised_nottoggled(p_obj_type)
            supervised_filled = self.check_supervised_filled(p_obj_type)
            supervised_notfilled = self.check_supervised_notfilled(p_obj_type)
            supervised_dirty = self.check_supervised_dirty(p_obj_type)
            supervised_notdirty = self.check_supervised_notdirty(p_obj_type)
            if supervised_open:
                if f'supervised_opened_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_opened_{p_obj_type}()')
                    current_goal = PddlParser.get_goal()
                    assert current_goal is not None
                    new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (open ?o1)',
                                                    f'(forall (?o1 - {p_obj_type}) (and (not (open ?o1))')
                    PddlParser.set_goal(new_goal)

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]
            if supervised_closed:
                if f'supervised_notopened_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_notopened_{p_obj_type}()')

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]
            if supervised_toggled:
                if f'supervised_toggled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_toggled_{p_obj_type}()')
                    current_goal = PddlParser.get_goal()
                    assert current_goal is not None
                    new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (toggled ?o1)',
                                                    f'(forall (?o1 - {p_obj_type}) (and (not (toggled ?o1))')
                    PddlParser.set_goal(new_goal)

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]
            if supervised_nottoggled:
                if f'supervised_nottoggled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_nottoggled_{p_obj_type}()')

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]
            if supervised_filled:
                if f'supervised_filled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_filled_{p_obj_type}()')
                    current_goal = PddlParser.get_goal()
                    assert current_goal is not None
                    new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (filled ?o1)',
                                                    f'(forall (?o1 - {p_obj_type}) (and (not (filled ?o1))')
                    PddlParser.set_goal(new_goal)

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]
            if supervised_notfilled:
                if f'supervised_notfilled_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_notfilled_{p_obj_type}()')

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]
            if supervised_dirty:
                if f'supervised_dirty_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_dirty_{p_obj_type}()')
                    current_goal = PddlParser.get_goal()
                    assert current_goal is not None
                    new_goal = current_goal.replace(f'(forall (?o1 - {p_obj_type}) (and (dirty ?o1)',
                                                    f'(forall (?o1 - {p_obj_type}) (and (not (dirty ?o1))')
                    PddlParser.set_goal(new_goal)

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]
            if supervised_notdirty:
                if f'supervised_notdirty_{p_obj_type}()' not in self.knowledge_base.all_predicates:
                    self.knowledge_base.add_predicate(f'supervised_notdirty_{p_obj_type}()')

                    self.knowledge_base.all_predicates = [p for p in self.knowledge_base.all_predicates
                                                          if not f'scanned({p_obj_type}' in p
                                                          and not f'manipulated({p_obj_type}' in p]

                    self.knowledge_base.self_supervised_predicates = [p for p in
                                                                      self.knowledge_base.self_supervised_predicates
                                                                      if not f'scanned({p_obj_type}' in p
                                                                      and not f'manipulated({p_obj_type}' in p]


    def check_supervised_open(self, obj_type):
        open_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('open')]
        open_supervised_objs = set([o for o in open_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == open_supervised_objs and len(all_obj_instances) > 0


    def check_supervised_close(self, obj_type):
        closed_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('not_open')]
        closed_supervised_objs = set([o for o in closed_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == closed_supervised_objs and len(all_obj_instances) > 0


    def check_supervised_toggled(self, obj_type):
        open_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('toggled')]
        open_supervised_objs = set([o for o in open_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == open_supervised_objs and len(all_obj_instances) > 0


    def check_supervised_nottoggled(self, obj_type):
        closed_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('not_toggled')]
        closed_supervised_objs = set([o for o in closed_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == closed_supervised_objs and len(all_obj_instances) > 0


    def check_supervised_filled(self, obj_type):
        open_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('filled')]
        open_supervised_objs = set([o for o in open_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == open_supervised_objs and len(all_obj_instances) > 0


    def check_supervised_notfilled(self, obj_type):
        closed_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('not_filled')]
        closed_supervised_objs = set([o for o in closed_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == closed_supervised_objs and len(all_obj_instances) > 0


    def check_supervised_dirty(self, obj_type):
        open_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('dirty')]
        open_supervised_objs = set([o for o in open_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == open_supervised_objs and len(all_obj_instances) > 0


    def check_supervised_notdirty(self, obj_type):
        closed_supervised_objs = [p['label'].split('(')[1].strip()[:-1]
                                    for p in self.knowledge_base.supervised_objects[obj_type]
                                    if p['label'].startswith('not_dirty')]
        closed_supervised_objs = set([o for o in closed_supervised_objs if f'scanned({o})' in self.knowledge_base.all_predicates])
        all_obj_instances = set([o['id'] for o in self.knowledge_base.all_objects[obj_type]
                                if self.knowledge_base.objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH])

        return all_obj_instances == closed_supervised_objs and len(all_obj_instances) > 0
