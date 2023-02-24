import copy
import datetime
import os

import numpy as np
from ai2thor.controller import Controller

import Configuration
from PAL.Learn.KnowledgeBase import KnowledgeManager
from PAL.Learn.Learner import Learner
from PAL.Learn.EnvironmentModels.State import State
from PAL.Plan.MetaPlanner import MetaPlanner
from Utils import Logger, PddlParser

import matplotlib.pyplot as plt

from Utils.ResultsPlotter import save_collected_data


class Agent:

    def __init__(self, scene="FloorPlan_Train1_1", position=None, init_rotation=None, init_horizon=None):

        # Set scene
        self.scene = scene

        # Initialize knowledge base
        self.knowledge_base = KnowledgeManager()

        # Initialize learner
        self.learner = Learner(self.knowledge_base)

        # Initialize meta planner
        self.meta_planner = MetaPlanner(self.knowledge_base)

        # Compute camera vertical field of view from horizontal field of view and camera frame height and width
        hfov = Configuration.FOV / 360. * 2. * np.pi
        vfov = 2. * np.arctan(np.tan(hfov / 2) * Configuration.FRAME_HEIGHT / Configuration.FRAME_WIDTH)
        vfov = np.rad2deg(vfov)

        # Initialize iTHOR simulator controller
        self.controller = Controller(renderDepthImage=True,
                                     renderObjectImage=True,
                                     visibilityDistance=Configuration.VISIBILITY_DISTANCE,
                                     gridSize=Configuration.MOVE_STEP,
                                     rotateStepDegrees=Configuration.ROTATION_STEP,
                                     scene=scene,
                                     # camera properties
                                     width=Configuration.FRAME_WIDTH,
                                     height=Configuration.FRAME_HEIGHT,
                                     fieldOfView=vfov,
                                     agentMode='default'
                                     )

        self.controller.step('SetRandomSeed', seed=0)  # invalid action in ai2thor 3.3.1

        # Move the agent in the initial position
        if position is not None or init_rotation is not None or init_horizon is not None:
            assert position is not None and init_rotation is not None and init_horizon is not None, \
                " If you do not want to use the default initial agent pose, you should set: initial position," \
                " rotation and horizon. See Agent.py constructor."

            self.controller.step(
                action="TeleportFull",
                position=position,
                rotation=dict(x=0, y=init_rotation, z=0),
                horizon=init_horizon,
                standing=True
            )

        # Set current iteration
        self.iter = 0

        # Initialize last (high-level) action effects
        self.last_action_effects = None

        # Initialize event (i.e. the environment observation after action execution)
        self.event = self.controller.step("Pass")

        # Perceive the environment
        perceptions = self.perceive()

        # Initialize initial state
        self.knowledge_base.current_state = self.create_state(perceptions)
        self.learner.add_current_state()

        # Create initial top view map
        self.learner.update_topview(os.path.join(Logger.LOG_DIR_PATH, "topview_{}.png".format(self.iter)),
                                    self.event.depth_frame,
                                    collision=True)

        # Flag indicating whether the goal has been achieved
        self.goal_achieved = False

        # Flag indicating whether the agent has collided with an obstacle while navigating the environment
        self.collision = False

        #
        self.hide_magnitude = 0


    def run(self, n_iter=Configuration.MAX_ITER):

        start = datetime.datetime.now()

        # Iterate for a maximum number of steps
        for i in range(n_iter):

            # Patch for 'explored' predicate in middle-size environments where map updates generate holes close to windows
            if i >= 50 and 'explored()' not in self.knowledge_base.all_predicates:
                self.knowledge_base.add_predicate('explored()')

            # Update PDDL state
            self.knowledge_base.update_pddl_state()

            # Save agent view image
            if Configuration.PRINT_CAMERA_VIEW_IMAGES:
                Logger.save_img("view_{}.png".format(i), self.event.frame)

            # Save agent depth view image
            if Configuration.PRINT_CAMERA_DEPTH_VIEW_IMAGES:
                Logger.save_img("depth_view_{}.png".format(i), (self.event.depth_frame/np.max(self.event.depth_frame)*255).astype('uint8'))

            # Check if goal has been achieved
            if self.goal_achieved:
                break

            # Set current iteration number
            self.iter = i

            # Compute a symbolic plan and get the first low-level action of the high-level action compilation
            event_action = self.meta_planner.plan()

            # If the maximum number of iterations has been reached, call 'Stop' action to end the episode
            if self.iter == Configuration.MAX_ITER - 1:
                event_action = "Stop"

            # DEBUG
            # if self.event_planner.subgoal is None:
            #     Logger.write('Event planner subgoal is EXPLORE()')
            # else:
            #     Logger.write('Event planner subgoal is {}'.format(self.event_planner.subgoal))

            Logger.write('{}:{} from state {} with {} self-supervised examples'.format(self.iter + 1,
                                                                                       event_action,
                                                                                       self.knowledge_base.current_state.id,
                                                                                       sum([len(v) for k,v in self.knowledge_base.supervised_objects.items()])))

            # Execute the chosen action
            self.event = self.step(event_action)
            # Necessary to sync unity window frame, otherwise it shows one step before
            self.controller.step("Pass")

            # Get feedback of action success
            last_action_success = self.event.metadata["lastActionSuccess"]

            # Patch for dirty object, when the agent tries to clean an object already cleaned, then the action execution
            # fails in the simulator, however we simulate its execution success.
            if event_action.lower().startswith('dirty') or event_action.lower().startswith('clean'):
                if "is already" in self.event.metadata['errorMessage']:
                    last_action_success = True

            # DEBUG
            if not last_action_success:
                Logger.write(f"Failed action message:{self.event.metadata['errorMessage']}")

            # Check collisions during navigation or rotation
            self.check_collision(event_action, last_action_success)

            # Remove old supervised predicates of manipulated object
            self.filter_self_supervised_preds(last_action_success, event_action)

            # Detect failure in following cases:
            # CASE 1: when manipulating objects (and remove them from the knowledge base)
            # CASE 2: when inspecting an object, a failure occurs whether the agent cannot see the inspected object
            # from a number n of states which are close to the inspected object (and while the agent is looking to the
            # inspected object direction)
            if ((event_action.startswith("Pick") or event_action.startswith("Put")
                or event_action.startswith("Open") or event_action.startswith("Close")
                or event_action.startswith("Toggle") or event_action.startswith("FillObjectWith")
                 or event_action.startswith("EmptyLiquid") or event_action.startswith("DirtyObject")
                 or event_action.startswith("CleanObject"))
                    and not last_action_success) \
                    or (len(self.meta_planner.event_planner.action_compiler.explored_goal_cells) >= Configuration.MAX_EXPLORED_GOAL_CELLS_INSPECT
                        and self.meta_planner.event_planner.subgoal is not None and self.meta_planner.event_planner.subgoal.lower().startswith('inspect')):
                Logger.write("NOT successfully executed action: {}".format(self.meta_planner.event_planner.subgoal))

                # Remove the object from the agent knowledge base
                removed_obj_id = self.meta_planner.event_planner.subgoal.lower().split('(')[1].strip()[:-1].split(',')[-1]
                self.learner.knowledge_base.remove_object(removed_obj_id)

                # Reset the event plan in the event planner, since the event planner subgoal is failed
                self.invalidate_plan()

            # Check if the PDDL action has been successfully executed, i.e., if the sequence of low-level actions
            # (aka 'event_plan') associated with the PDDL action execution has been entirely executed.
            elif last_action_success and self.meta_planner.event_planner.event_plan is not None \
                    and len(self.meta_planner.event_planner.event_plan) == 0 and self.meta_planner.event_planner.subgoal is not None:

                # DEBUG
                Logger.write("Successfully executed action: {}".format(self.meta_planner.event_planner.subgoal))

                # Get successfully executed pddl action effects
                pddl_action = self.meta_planner.event_planner.subgoal.lower()
                self.last_action_effects = self.get_action_effects(self.meta_planner.event_planner.subgoal)

                # Reset useless_goal_cells used by 'Inspect' PDDL action
                self.meta_planner.event_planner.action_compiler.explored_goal_cells = []

                # Check if the PDDL goal has been achieved
                if self.meta_planner.event_planner.subgoal == 'STOP()':
                    self.goal_achieved = True

                # TMP PATCH UPDATE OPENED/CLOSED OBJECT
                if self.meta_planner.event_planner.subgoal.lower().startswith('open') \
                        or self.meta_planner.event_planner.subgoal.lower().startswith('close'):

                    # DEBUG
                    print('UPDATING CLOSED/OPENED OBJECT POSITION')

                    manipulated_object = self.meta_planner.event_planner.subgoal.lower().strip().split('(')[1][:-1]
                    self.learner.update_opened_object(manipulated_object, self.event.depth_frame)

                # Invalidate the PDDL plan to perform replanning
                self.invalidate_plan()

            # Perceive the environment
            perceptions = self.perceive()

            # Add a new state to the finite state machine created by the agent
            new_state = self.create_state(perceptions)
            self.learner.add_state(new_state)

            # If a *TRUSTED* PDDL action has been successfully executed, apply action effects
            if self.last_action_effects is not None:
                self.apply_action_effects(new_state, pddl_action)
                self.last_action_effects = None
                pddl_action = None

            # Add self-supervised labels to the current state
            new_state.self_supervised_predicates = self.get_self_supervised_predicates(new_state)
            self.learner.update_supervised_objects(new_state)

            # Add transition to the finite state machine created by the agent
            self.learner.add_transition(self.knowledge_base.current_state, event_action, new_state)

            # Update current state in the knowledge base
            self.knowledge_base.current_state = new_state

            # Update top view map <==> the agent has collided
            self.learner.update_topview(os.path.join(Logger.LOG_DIR_PATH, "topview_{}.png".format(self.iter)),
                                        self.event.depth_frame,
                                        collision=self.collision)
            self.collision = False

            # (DEBUG) Store collected data for training the property classifiers
            # if i > 0 and (i + 1) % 100 == 0:
            #     save_collected_data(self.knowledge_base, f"{self.scene}_{i+1}steps")

        # if self.goal_achieved:
        #     Logger.write('Episode succeeds.')
        # else:
        #     Logger.write('Episode fails.')

        # DEBUG
        end = datetime.datetime.now()
        Logger.write("Episode computational time: {} seconds".format((end-start).seconds))

        # Store collected data for training the property classifiers
        save_collected_data(self.knowledge_base, self.scene)

        # Release resources
        self.controller.stop()
        plt.close(self.knowledge_base.map_model.fig)
        Logger.LOG_FILE.close()


    def invalidate_plan(self):
        # Reset event plan in event planner since the event planner subgoal is failed
        self.meta_planner.event_planner.subgoal = None
        self.meta_planner.event_planner.event_plan = None
        self.meta_planner.event_planner.path_plan = None
        self.meta_planner.pddl_plan = None
        self.meta_planner.event_planner.action_compiler.explored_goal_cells = []


    def check_op_eff_consistency(self, new_state, ground_pddl_action):
        pddl_operator = ground_pddl_action.split('(')[0].lower()

        if pddl_operator == 'get_close_and_look_at_openable':
            action_obj_id = ground_pddl_action.split('(')[1].strip().lower()[:-1]
            action_obj_type = action_obj_id.split('_')[0]

            if action_obj_type not in new_state.visible_objects.keys() \
                    or action_obj_id not in [o['id'] for o in new_state.visible_objects[action_obj_type]]:

                # 1) Check if inspected predicated is still true for some other state
                self.check_inspected_predicate(action_obj_id, new_state)

                # 2) Invalidate plan
                self.invalidate_plan()
                Logger.write(f'WARNING: cannot see expected object {action_obj_id}, removing old state.')

                return False
        else:
            Logger.write(f"ERROR: you should implement how to check consistency of the effects of {pddl_operator}, look"
                         f" at check_op_eff_consistency in Agent.py")
            exit()

        return True

    def check_inspected_predicate(self, object_id, current_state):
        goal_object_states = [s for s in self.knowledge_base.fsm_model.states
                              if f"inspected({object_id})" in s.visible_predicates]
        # Remove inspected predicate from all states close to the current one and from knowledge base
        position_thrsh = 0.15
        agent_angle_thrsh = 5
        current_pos = np.array(current_state.perceptions['position'])
        current_angle = current_state.perceptions['agent_angle']
        removed_states = []
        for state in goal_object_states:
            if np.linalg.norm(current_pos - np.array(state.perceptions['position'])) < position_thrsh \
                    and current_angle - state.perceptions['agent_angle'] < agent_angle_thrsh:
                # state.visible_predicates
                removed_states.append(state)

        goal_object_states = [s for s in goal_object_states if s not in removed_states]
        self.knowledge_base.fsm_model.states = [s for s in self.knowledge_base.fsm_model.states
                                                if s not in removed_states]

        # Check if inspected predicate of action object has to be removed from knowledge base
        if len(goal_object_states) == 0:
            self.knowledge_base.remove_predicate(f'inspected({object_id})')


    def apply_action_effects(self, new_state, pddl_action):
        pddl_operator = pddl_action.split('(')[0].lower()
        consistency = True
        if pddl_operator in Configuration.UNTRUSTED_PDDL_OPERATORS:
            consistency = self.check_op_eff_consistency(new_state, pddl_action)

        if consistency:
            # DEBUG
            print(f"INFO: Applying action effects: {self.last_action_effects}")

            # Update pddl state
            pos_effect = [e for e in self.last_action_effects if not e.startswith("(not ")]
            neg_effect = [e.replace('(not ', '').strip()[:-1] for e in self.last_action_effects if
                          e.startswith("(not ")]
            [self.knowledge_base.add_predicate(e) for e in pos_effect]
            [self.knowledge_base.remove_predicate(e) for e in neg_effect]

            # self.knowledge_base.update_pddl_state()

            # Add self-supervised labels to the knowledge base
            [self.knowledge_base.self_supervised_predicates.append(e) for e in pos_effect
             if e not in self.knowledge_base.self_supervised_predicates]
            [self.knowledge_base.self_supervised_predicates.remove(e) for e in neg_effect
             if e in self.knowledge_base.self_supervised_predicates]
            [self.knowledge_base.self_supervised_predicates.append("not_" + e) for e in neg_effect
             if "not_" + e not in self.knowledge_base.self_supervised_predicates]
            [self.knowledge_base.self_supervised_predicates.remove("not_" + e) for e in pos_effect
             if "not_" + e in self.knowledge_base.self_supervised_predicates]


    def check_collision(self, event_action, last_action_success):

        # Detect collision when moving forward (and eventually update occupancy map)
        if event_action == "MoveAhead" and not last_action_success:
            Logger.write("Collision detected")
            self.update_collision_map()
            self.meta_planner.event_planner.path_plan = None
            self.meta_planner.event_planner.event_plan = None
            self.collision = True

        # Detect collision when rotating (and eventually change rotation direction)
        if event_action.startswith("Rotate") and not last_action_success:
            Logger.write("Collision detected")
            self.meta_planner.event_planner.path_plan = None
            self.meta_planner.event_planner.event_plan = None
            self.meta_planner.event_planner.action_compiler.rotation_collision = True

            # This is necessary for 'LOOK_AT' symbolic action
            if self.meta_planner.event_planner.action_compiler.rotate_dir == Configuration.ROTATE_RIGHT:
                self.meta_planner.event_planner.action_compiler.rotate_dir = Configuration.ROTATE_LEFT
            else:
                self.meta_planner.event_planner.action_compiler.rotate_dir = Configuration.ROTATE_RIGHT


    def filter_self_supervised_preds(self, last_action_success, event_action):
        event_action = event_action.lower()
        if last_action_success and event_action.startswith('close') or event_action.startswith('open'):
            manipulated_obj_id = self.meta_planner.event_planner.subgoal.lower().split('(')[1][:-1]
            manipulated_obj_type = manipulated_obj_id.split('_')[0]
            manipulated_obj = [o for o in self.knowledge_base.all_objects[manipulated_obj_type]
                               if o['id'] == manipulated_obj_id][0]
            manipulated_obj_pos = [manipulated_obj['map_x'], manipulated_obj['map_y'], manipulated_obj['map_z']]

            # Remove self supervised predicates of the manipulated object
            self.knowledge_base.self_supervised_predicates = [p for p in self.knowledge_base.self_supervised_predicates
                                                              if f"{manipulated_obj_id}," not in p and f"{manipulated_obj_id})" not in p]

            obj_close_to_manipulated = [o['id'] for o in self.knowledge_base.all_objects[manipulated_obj_type]
                                        if np.linalg.norm(np.array([o['map_x'], o['map_y'], o['map_z']]) - np.array(manipulated_obj_pos)) < 0.5]

            removed_self_supervised_predicates = list(set([p for p in self.knowledge_base.self_supervised_predicates
                                                  for near_obj in obj_close_to_manipulated
                                                  if f"{near_obj}," in p or f"{near_obj})" in p]))
            self.knowledge_base.self_supervised_predicates = [p for  p in self.knowledge_base.self_supervised_predicates
                                                              if p not in removed_self_supervised_predicates]


    def get_self_supervised_predicates(self, state):

        # Add self-supervised labels to the current state
        self_supervised_predicates = []
        state_objs = []
        for obj_type, obj_insts in state.visible_objects.items():
            [state_objs.append(obj_inst['id'].lower()) for obj_inst in obj_insts]

        for supervised_pred in self.knowledge_base.self_supervised_predicates:
            pred_objs = [o.lower() for o in supervised_pred.split('(')[1].strip()[:-1].split(',')]
            if set(pred_objs).issubset(set(state_objs)):
            # if set(pred_objs).issubset(set(state_objs)) or supervised_pred.startswith('scanned('):
                self_supervised_predicates.append(supervised_pred)

        # Add self supervised predicates
        return self_supervised_predicates


    def step(self, action):
        action_result = None

        if action.startswith("Rotate") or action.startswith("Look"):
            if len(action.split("|")) > 1:
                degrees = round(float(action.split("|")[1]), 1)
                action_result = self.controller.step(action=action.split("|")[0], degrees=degrees)
            else:
                action_result = self.controller.step(action=action)

        elif action.startswith("FillObject"):
            # If xy camera coordinates are used to perform the action, i.e., are in the action name
            if len(action.split("|")) > 2:
                x_pos = round(float(action.split("|")[1]), 2)
                y_pos = round(float(action.split("|")[2]), 2)

                action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos, fillLiquid='water')

        elif action.startswith("OpenObject") or action.startswith("CloseObject") or action.startswith("ToggleObject") \
                or action.startswith("EmptyLiquid") or action.startswith("DirtyObject") or action.startswith("CleanObject"):
            # If xy camera coordinates are used to perform the action, i.e., are in the action name
            if len(action.split("|")) > 2:
                x_pos = round(float(action.split("|")[1]), 2)
                y_pos = round(float(action.split("|")[2]), 2)

                action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

        elif action.startswith("PickupObject") or action.startswith("PutObject"):
            # Store held objects
            old_inventory = copy.deepcopy(self.event.metadata['inventoryObjects'])

            # If xy camera coordinates are used to perform the action, i.e., are in the action name
            if len(action.split("|")) > 2:
                x_pos = round(float(action.split("|")[1]), 2)
                y_pos = round(float(action.split("|")[2]), 2)

                # The held object is hidden, however the simulator persistently sees it, hence if the target point
                # overlaps with the held (and hidden) object, the 'PutObject' action fails
                # ==> move up/down the held object to avoid overlapping with target point of 'PutObject' action.
                if action.startswith("PutObject"):
                    if y_pos >= 0.5:
                        move_magnitude = 0.7
                        moved = False
                        while not moved and move_magnitude > 0:
                            moved = self.controller.step("MoveHandUp", moveMagnitude=move_magnitude).metadata['lastActionSuccess']
                            move_magnitude -= 0.1
                        action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

                        # Check whether the object is within the visibility distance but cannot be placed into
                        # the container one due to object categories constraints
                        if 'cannot be placed in' in action_result.metadata['errorMessage']:
                            action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos, forceAction=True)

                        if moved:
                            self.controller.step("MoveHandDown", moveMagnitude=move_magnitude)

                    else:
                        move_magnitude = 0.7
                        moved = False
                        while not moved and move_magnitude > 0:
                            moved = self.controller.step("MoveHandDown", moveMagnitude=move_magnitude).metadata['lastActionSuccess']
                            move_magnitude -= 0.1
                        action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

                        # Check whether the object is within the visibility distance but cannot be placed into
                        # the container one due to object categories constraints
                        if 'cannot be placed in' in action_result.metadata['errorMessage']:
                            action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos, forceAction=True)

                        if moved:
                            self.controller.step("MoveHandUp", moveMagnitude=move_magnitude)
                else:
                    action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

                # Hide picked up objects
                if Configuration.HIDE_PICKED_OBJECTS and action.startswith("PickupObject") \
                        and action_result.metadata['lastActionSuccess']:
                    for picked_obj in action_result.metadata['inventoryObjects']:
                        self.hide_magnitude = 0.7
                        debug = self.controller.step("MoveHandDown", moveMagnitude=self.hide_magnitude)
                        while not debug.metadata['lastActionSuccess'] and self.hide_magnitude > 0:
                            self.hide_magnitude -= 0.1
                            debug = self.controller.step("MoveHandDown", moveMagnitude=self.hide_magnitude)

                        self.controller.step('HideObject', objectId=picked_obj['objectId'])
                        action_result = self.controller.step('Pass')

                # Unhide put down objects
                elif Configuration.HIDE_PICKED_OBJECTS and action.startswith("PutObject") \
                        and action_result.metadata['lastActionSuccess']:
                    for released_obj in old_inventory:
                        self.controller.step('UnhideObject', objectId=released_obj['objectId'])
                    if self.hide_magnitude > 0:
                        debug = self.controller.step("MoveHandUp", moveMagnitude=self.hide_magnitude)
                        if not debug.metadata['lastActionSuccess']:
                            Logger.write('WARNING: MoveHandUp failed after putting object down. Check step() function'
                                         'in Agent.py.')
                    action_result = self.controller.step('Pass')

            # Other cases
            else:
                print('ERROR: you should manage the case where a pickup/putdown action is performed without '
                      'passing input xy camera coordinates. Look at step() method in Agent.py .')
                exit()

        elif action == "Stop":
            action_result = self.step("Pass")
            self.goal_achieved = True

        else:
            # Execute "move" action in the environment
            action_result = self.controller.step(action=action)

        self.knowledge_base.update_all_obj_position(action, action_result, self.meta_planner.event_planner.subgoal)

        return action_result


    def perceive(self):

        # Get perceptions
        x_pos = self.event.metadata['agent']['position']['x']
        y_pos = self.event.metadata['agent']['position']['z']
        camera_z_pos = self.event.metadata['cameraPosition']['y']
        angle = (360 - self.event.metadata['agent']['rotation']['y'] + 90) % 360
        camera_angle = -int(round(self.event.metadata['agent']['cameraHorizon']))  # tilt angle of the camera
        rgb_img = self.event.frame
        depth_img = self.event.depth_frame

        perceptions = {
            'position': [x_pos, y_pos, camera_z_pos],
            'agent_angle': angle,
            'camera_tilt': camera_angle,
            'rgb': rgb_img,
            'depth': depth_img
        }

        return perceptions


    def create_state(self, perceptions):

        rgb_img = perceptions['rgb']
        depth_img = perceptions['depth']
        agent_pos = perceptions['position']
        angle = perceptions['agent_angle']

        # Predict visible objects in agent view
        visible_objects = self.learner.get_visible_objects(rgb_img, depth_img, agent_pos, angle, self.event)

        # Update overall knowledge about objects
        visible_objects = self.knowledge_base.update_objects(visible_objects, agent_pos)

        # Predict predicates about visible objects
        visible_predicates = self.learner.scene_classifier.get_visible_predicates(visible_objects, rgb_img)

        # Remove inconsistent predictions with self-supervisions
        visible_predicates = [p for p in visible_predicates if not f"not_{p}" in self.knowledge_base.self_supervised_predicates]

        # Get visible object relationships and update predicates
        self.update_predicates(visible_predicates)

        # Create new state
        new_id = max([s.id for s in self.knowledge_base.fsm_model.states], default=0) + 1
        s_new = State(new_id, perceptions, visible_objects, visible_predicates, self.controller.last_event)

        return s_new


    def update_predicates(self, visible_predicates):

        # Update overall knowledge about predicates
        self.knowledge_base.update_all_predicates(visible_predicates)

        # Update pddl state
        self.knowledge_base.update_pddl_state()


    def update_collision_map(self):

        agent_theta = self.knowledge_base.current_state.perceptions['agent_angle']
        agent_position = self.knowledge_base.current_state.perceptions['position']

        # Map agent position into grid
        start = [agent_position[0]*100, agent_position[1]*100]
        start_grid = (int(round((start[0] - self.knowledge_base.map_model.x_min) / self.knowledge_base.map_model.dx)),
                      int(round((start[1] - self.knowledge_base.map_model.y_min) / self.knowledge_base.map_model.dy)))
        # start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid
        collision_cell = None

        NOISE_THRSH = 30  # noise threshold between two subsequent cells
        if 0 - NOISE_THRSH <= agent_theta <= 0 + NOISE_THRSH or 360 - NOISE_THRSH < agent_theta <= 360 + NOISE_THRSH:
            collision_cell = [start_grid[1], start_grid[0] + 1]
        elif NOISE_THRSH < agent_theta <= 90 - NOISE_THRSH:
            collision_cell = [start_grid[1] + 1, start_grid[0] + 1]
        elif 90 - NOISE_THRSH < agent_theta <= 90 + NOISE_THRSH:
            collision_cell = [start_grid[1] + 1, start_grid[0]]
        elif 90 + NOISE_THRSH < agent_theta <= 180 - NOISE_THRSH:
            collision_cell = [start_grid[1] + 1, start_grid[0] - 1]
        elif 180 - NOISE_THRSH < agent_theta <= 180 + NOISE_THRSH:
            collision_cell = [start_grid[1], start_grid[0] - 1]
        elif 180 + NOISE_THRSH < agent_theta <= 270 - NOISE_THRSH:
            collision_cell = [start_grid[1] - 1, start_grid[0] - 1]
        elif 270 - NOISE_THRSH < agent_theta <= 270 + NOISE_THRSH:
            collision_cell = [start_grid[1] - 1, start_grid[0]]
        elif 270 + NOISE_THRSH < agent_theta <= 360 - NOISE_THRSH:
            collision_cell = [start_grid[1] - 1, start_grid[0] + 1]

        assert collision_cell is not None, "Cannot add null collision cell"

        self.knowledge_base.map_model.collision_cells.append(collision_cell)

        # Check if the goal position on the grid is the same as the collision cell just added. If this is the case,
        # change the goal position in the path planner.
        goal = [self.meta_planner.event_planner.action_compiler.path_planner.goal_position[0] * 100,
                self.meta_planner.event_planner.action_compiler.path_planner.goal_position[1] * 100]
        goal_grid = [int(round((goal[1]-self.knowledge_base.map_model.y_min)/self.knowledge_base.map_model.dy)),
                     int(round((goal[0]-self.knowledge_base.map_model.x_min)/self.knowledge_base.map_model.dx))]

        if goal_grid == collision_cell:
            print('[DEBUG] Changing goal position')

            self.goal_position = [(Configuration.MAP_X_MIN + (Configuration.MOVE_STEP * 100 * 2)) / 100,
                                  (Configuration.MAP_Y_MIN + (Configuration.MOVE_STEP * 100 * 2)) / 100]


    def get_action_effects(self, action_name):

        last_action_effects = []

        # Get operator name
        op_name = action_name.split("(")[0].strip().lower()

        # Get operator objects
        op_objs = {"?param_{}".format(i + 1): obj
                   for i, obj in enumerate(action_name.split("(")[1][:-1].strip().lower().split(","))}

        # Get operator effects
        op_effects = PddlParser.get_operator_effects(op_name)

        # Update predicates with positive effects
        for pred in [pred for pred in op_effects if not pred.startswith("(not ")]:
            # Replace predicate variables with pddl action input objects
            for k, v in op_objs.items():
                pred = pred.replace(k, v)
            # Get predicate name and grounded input objects
            pred_name = pred[1:-1].split()[0].strip()
            pred_objs = [obj.strip() for obj in pred[1:-1].split()[1:]]
            pred_renamed = "{}({})".format(pred_name, ",".join(pred_objs))
            # if pred_renamed not in self.knowledge_base.all_predicates:
            last_action_effects.append(pred_renamed)

        # Update predicates with negative effects
        for pred in [pred for pred in op_effects if pred.startswith("(not ")]:
            pred = pred.replace("(not ", "").strip()[:-1].strip()
            # Replace predicate variables with pddl action input objects
            for k, v in op_objs.items():
                pred = pred.replace(k, v)
            # Get predicate name and grounded input objects
            pred_name = pred[1:-1].split()[0].strip()
            pred_objs = [obj.strip() for obj in pred[1:-1].split()[1:]]
            pred_renamed = "{}({})".format(pred_name, ",".join(pred_objs))
            if pred_renamed in self.knowledge_base.all_predicates:
                last_action_effects.append("(not {})".format(pred_renamed))

        return last_action_effects
