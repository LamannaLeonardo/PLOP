import copy
import math
import random

import numpy as np

import Configuration
from PAL.Plan.PathPlanner import PathPlanner
from Utils import Logger
from Utils.depth_util import get_xyz_points_matrix_from_depth


class ActionCompiler:

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

        # Set path planner
        self.path_planner = PathPlanner(self.knowledge_base)

        # Goal cells already explored when executing the action 'INSPECT', i.e., goal cells from which the goal
        # object is not recognized.
        self.explored_goal_cells = []

        # Flag that indicates whether the agent has collided during rotation with an held object
        self.rotation_collision = False

        # Rotate direction when moving agents
        self.rotate_dir = Configuration.ROTATE_DIRECTION


    def compile(self, symbolic_action):

        if symbolic_action.lower().strip().split('(')[0] == 'open':
            return self.compile_open(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0] == 'close':
            return self.compile_close(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0] == 'inspect':
            return self.compile_inspect(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('scan'):
            return self.compile_scan(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('get_close_and_look_at'):
            return self.compile_get_close(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('toggle_on'):
            return self.compile_toggle_on(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('toggle_off'):
            return self.compile_toggle_off(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('fill'):
            return self.compile_fill(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('unfill'):
            return self.compile_unfill(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('dirty'):
            return self.compile_dirty(symbolic_action)

        elif symbolic_action.lower().strip().split('(')[0].startswith('clean'):
            return self.compile_clean(symbolic_action)

        else:
            Logger.write("ERROR: symbolic action {} is not implemented in ActionCompiler.py".format(symbolic_action))
            exit()


    def compile_open(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=True)

        event_plan = ["OpenObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_toggle_on(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=True)

        event_plan = ["ToggleObjectOn|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_toggle_off(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=True)

        event_plan = ["ToggleObjectOff|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_fill(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=False)

        event_plan = ["FillObjectWithLiquid|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_dirty(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=False)

        event_plan = ["DirtyObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_clean(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=False)

        event_plan = ["CleanObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_unfill(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=False)

        event_plan = ["EmptyLiquidFromObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_close(self, symbolic_action):
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id, noise=True)

        event_plan = ["CloseObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

        return event_plan


    def compile_inspect(self, symbolic_action):

        # Get agent position
        agent_x = self.knowledge_base.current_state.perceptions['position'][0]
        agent_y = self.knowledge_base.current_state.perceptions['position'][1]

        # Get goal object position
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()
        goal_object_type = goal_object_id.split("_")[0].strip()
        goal_object = [obj for obj in self.knowledge_base.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]
        goal_obj_x = goal_object['map_x']
        goal_obj_y = goal_object['map_y']
        goal_obj_z = goal_object['map_z']
        goal_position = [goal_obj_x, goal_obj_y, goal_obj_z]
        start_position = {'x': agent_x, 'y': agent_y}

        # Plan to reach a position close to the goal object
        event_plan = self.path_planner.path_planning_greedy(start_position, goal_position,
                                                            around_goal_distance=Configuration.CLOSE_TO_OBJ_DISTANCE,
                                                            non_goal_grid_cells=self.explored_goal_cells)
        if event_plan is None:
            Logger.write('WARNING: goal object is not reachable, deleting it from the agent knowledge.')
            return

        # Append a fictitious action to see last event plan action results before confirming subgoal success
        if 'fictitious action' not in event_plan:
            event_plan.append('fictitious action')

        # Adjust plan whether the agent has collided while rotating and holding an object
        if self.rotation_collision and len([a for a in event_plan if a != 'fictitious action']) > 0:
            event_plan = self.adjust_event_plan_rotations(event_plan)
            self.rotation_collision = False

        # If the agent has reached a position close to the goal object, look at the goal object
        if len(event_plan) == 1:

            if not 'fictitious action' in event_plan:
                Logger.write('ERROR: look at compile_inspect() in ActionCompiler.py')
                exit()
            event_plan = []

            # Get angle between agent and goal object position
            agent_angle = int(round(self.knowledge_base.current_state.perceptions['agent_angle']))  # rescale agent angle according to ref sys
            agent2obj_angle = int(
                (math.degrees(math.atan2(goal_obj_y - agent_y, goal_obj_x - agent_x)) - agent_angle)) % 360

            rotate_right = agent2obj_angle >= 180

            # Set angle according to agent rotation direction (left or right)
            if rotate_right:
                agent2obj_angle = 360 - agent2obj_angle

            while agent2obj_angle not in range(int(-Configuration.FOV / 2), int(Configuration.FOV / 2) + 1) \
                    and agent2obj_angle not in range(int(360 - Configuration.FOV / 2),
                                                     int(360 + Configuration.FOV / 2) + 1):

                if not rotate_right:
                    event_plan.append("RotateLeft")
                    agent2obj_angle -= Configuration.ROTATION_STEP
                else:
                    event_plan.append("RotateRight")
                    agent2obj_angle -= Configuration.ROTATION_STEP

            # Adjust plan whether the agent has collided while rotating and holding an object
            if self.rotation_collision:
                event_plan = self.adjust_event_plan_rotations(event_plan)
                self.rotation_collision = False

            camera_angle = int(self.knowledge_base.current_state.perceptions['camera_tilt'])
            agent2obj_z_angle = int((math.degrees(math.atan2((goal_obj_z) - Configuration.CAMERA_HEIGHT,
                                                             (goal_obj_y - agent_y)**2 + (goal_obj_x - agent_x)**2)) - camera_angle))

            # Adjust camera inclination to better see the object
            if agent2obj_z_angle < -15 and camera_angle - 30 >= -Configuration.MAX_CAM_ANGLE*2:  # Assume an inclination of 30 degrees
                look_down = []
                look_up = []
                next_camera_angle = camera_angle
                while agent2obj_z_angle < -15 and next_camera_angle - 30 >= -Configuration.MAX_CAM_ANGLE*2:
                    look_down.append("LookDown")  # Assume an inclination of 30 degrees
                    agent2obj_z_angle += 30
                    next_camera_angle -= 30
                event_plan = event_plan + look_down + look_up

            elif agent2obj_z_angle > 15 and camera_angle + 30 <= Configuration.MAX_CAM_ANGLE:  # Assume an inclination of 30 degrees
                look_down = []
                look_up = []
                next_camera_angle = camera_angle
                while agent2obj_z_angle > 15 and next_camera_angle + 30 <= Configuration.MAX_CAM_ANGLE:
                    look_up.append("LookUp")  # Assume an inclination of 30 degrees
                    agent2obj_z_angle -= 30
                    next_camera_angle += 30
                event_plan = event_plan + look_up + look_down

            # Append a fictitious action to see last event plan action results before confirming subgoal success
            if 'fictitious action' not in event_plan:
                event_plan.append('fictitious action')

        return event_plan


    def compile_scan(self, symbolic_action):

        # Get agent position
        agent_x = self.knowledge_base.current_state.perceptions['position'][0]
        agent_y = self.knowledge_base.current_state.perceptions['position'][1]

        # Get goal object position
        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()
        goal_object_type = goal_object_id.split("_")[0].strip()
        goal_object = [obj for obj in self.knowledge_base.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]
        goal_obj_x = goal_object['map_x']
        goal_obj_y = goal_object['map_y']
        goal_obj_z = goal_object['map_z']
        goal_position = [goal_obj_x, goal_obj_y, goal_obj_z]
        start_position = {'x': agent_x, 'y': agent_y}

        event_plan = self.path_planner.path_planning_greedy(start_position, goal_position,
                                                            around_goal_distance=Configuration.CLOSE_TO_OBJ_DISTANCE * 4,
                                                            non_goal_grid_cells=self.explored_goal_cells)
        if event_plan is None:
            Logger.write('WARNING: goal object is not reachable, deleting it from the agent knowledge.')
            return

        # Append a fictitious action to see last event plan action results before confirming subgoal success
        if 'fictitious action' not in event_plan:
            event_plan.append('fictitious action')

        # Adjust plan whether the agent has collided while rotating and holding an object
        if self.rotation_collision and len([a for a in event_plan if a != 'fictitious action']) > 0:
            event_plan = self.adjust_event_plan_rotations(event_plan)
            self.rotation_collision = False

        # If the agent has reached a position close to the goal object, look at the goal object
        if len(event_plan) == 1:

            if not 'fictitious action' in event_plan:
                Logger.write('ERROR: look at compile_scan() in ActionCompiler.py')
                exit()
            event_plan = []

            # Get angle between agent and goal object position
            agent_angle = int(round(self.knowledge_base.current_state.perceptions['agent_angle']))  # rescale agent angle according to ref sys
            agent2obj_angle = int(
                (math.degrees(math.atan2(goal_obj_y - agent_y, goal_obj_x - agent_x)) - agent_angle)) % 360

            rotate_right = agent2obj_angle >= 180

            # Set angle according to agent rotation direction (left or right)
            if rotate_right:
                agent2obj_angle = 360 - agent2obj_angle

            while agent2obj_angle not in range(int(-Configuration.FOV / 2), int(Configuration.FOV / 2) + 1) \
                    and agent2obj_angle not in range(int(360 - Configuration.FOV / 2),
                                                     int(360 + Configuration.FOV / 2) + 1):

                if not rotate_right:
                    event_plan.append("RotateLeft")
                    agent2obj_angle -= Configuration.ROTATION_STEP
                else:
                    event_plan.append("RotateRight")
                    agent2obj_angle -= Configuration.ROTATION_STEP

            # Adjust plan whether the agent has collided while rotating and holding an object
            if self.rotation_collision:
                event_plan = self.adjust_event_plan_rotations(event_plan)
                self.rotation_collision = False

            # camera_angle = -int(self.knowledge_base.current_state.perceptions['camera_tilt'])
            camera_angle = int(self.knowledge_base.current_state.perceptions['camera_tilt'])
            agent2obj_z_angle = int((math.degrees(math.atan2((goal_obj_z) - Configuration.CAMERA_HEIGHT,
                                                             (goal_obj_y - agent_y)**2 + (goal_obj_x - agent_x)**2)) + camera_angle))

            if agent2obj_z_angle < -30 and camera_angle - 30 >= -Configuration.MAX_CAM_ANGLE:  # Assume an inclination of 30 degrees
                look_down = []
                look_up = []
                while agent2obj_z_angle < -30:
                    look_down.append("LookDown")  # Assume an inclination of 30 degrees
                    agent2obj_z_angle += 30
                event_plan = event_plan + look_down + look_up

            elif agent2obj_z_angle > 30 and camera_angle + 30 < Configuration.MAX_CAM_ANGLE:  # Assume an inclination of 30 degrees
                look_down = []
                look_up = []
                while agent2obj_z_angle > 30:
                    look_up.append("LookUp")  # Assume an inclination of 30 degrees
                    agent2obj_z_angle -= 30
                event_plan = event_plan + look_up + look_down

            # Append a fictitious action to see last event plan action results before confirming event planner
            # subgoal success
            if 'fictitious action' not in event_plan:
                event_plan.append('fictitious action')

        if len(self.explored_goal_cells) >= Configuration.MAX_EXPLORED_GOAL_CELLS_SCAN - 1:
            event_plan = [e for e in event_plan if e != 'fictitious action']

        return event_plan


    def compile_get_close(self, symbolic_action):

        goal_object_id = symbolic_action.split("(")[1].strip()[:-1].lower()

        # Choose the state which minimizes the distance from the goal object and set the state position as a goal one
        goal_pose = self.set_goal_obj_state_position(goal_object_id)

        if goal_pose is None:
            self.explored_goal_cells = list(range(Configuration.MAX_EXPLORED_GOAL_CELLS_INSPECT))
            return None

        self.path_planner.goal_position = goal_pose[:-2]  # last two numbers are agent and cam rotations

        event_plan = self.path_planner.path_planning()

        # Adjust plan whether the agent has collided while rotating and holding an object
        if self.rotation_collision and event_plan is not None:
            event_plan = self.adjust_event_plan_rotations(event_plan)
            self.rotation_collision = False

        # Check if goal object is still reachable. E.g., if the agent collides while holding an other object
        # there could be no more path towards the goal object, hence it is removed from the knowledge base.
        if event_plan is None:
            self.explored_goal_cells = list(range(Configuration.MAX_EXPLORED_GOAL_CELLS_INSPECT))
            Logger.write('WARNING: the goal object cannot be reached. It is likely that the agent has manipulated '
                         'objects which obstruct the path.')
            return None

        event_plan.append('fictitious_action')

        if len(event_plan) == 1:
            event_plan = []

            goal_agent_angle = goal_pose[2]
            goal_camera_tilt = goal_pose[3]
            current_agent_angle = self.knowledge_base.current_state.perceptions['agent_angle']
            current_camera_tilt = self.knowledge_base.current_state.perceptions['camera_tilt']

            delta_agent_angle = int(round((goal_agent_angle - current_agent_angle) % 360))
            # Optimize rotations
            if delta_agent_angle > 180:
                delta_agent_angle = delta_agent_angle - 360
            delta_camera_tilt = int(round((goal_camera_tilt - current_camera_tilt)))

            # Rotate agent
            if delta_agent_angle > 0:
                [event_plan.append('RotateLeft') for _ in range(delta_agent_angle // Configuration.ROTATION_STEP)]
            else:
                [event_plan.append('RotateRight') for _ in range(delta_agent_angle // -Configuration.ROTATION_STEP)]

            # Rotate camera
            if delta_camera_tilt > 0:
                [event_plan.append('LookUp') for _ in range(delta_camera_tilt // 30)]  # Assume an inclination of 30 degrees
            else:
                [event_plan.append('LookDown') for _ in range(delta_camera_tilt // -30)]  # Assume an inclination of 30 degrees

        return event_plan


    def get_obj_bb_centroid(self, goal_object_id, noise=False):

        # Get goal object information from current agent state
        goal_object_type = goal_object_id.split("_")[0].strip()
        goal_object = [obj for obj in self.knowledge_base.all_objects[goal_object_type]
                       if obj['id'] == goal_object_id][0]

        # Return object bbox center relative to agent view
        obj_bb_centroid = [goal_object['bb']['center'][0] / Configuration.FRAME_WIDTH,
                           goal_object['bb']['center'][1] / Configuration.FRAME_HEIGHT]

        if noise:
            add_noise_x = random.uniform(-0.02, 0.02)
            add_noise_y = random.uniform(-0.02, 0.02)
            obj_bb_centroid = [obj_bb_centroid[0] + add_noise_x, obj_bb_centroid[1] + add_noise_y]

        return obj_bb_centroid


# Adjust plan whether the agent has collided while rotating and holding an object
    def adjust_event_plan_rotations(self, event_plan):

        rotate_left = []
        rotate_right = []
        for action in event_plan:
            if action == 'RotateLeft':
                rotate_left.append('RotateLeft')
            elif action == 'RotateRight':
                rotate_right.append('RotateRight')
            else:
                break

        event_plan = event_plan[len(rotate_left) + len(rotate_right):]

        assert len(rotate_right) == 0 or len(rotate_left) == 0, 'Check adjust_event_plan_rotations() in ActionCompiler.py'

        complete_rotations = int(360 / Configuration.ROTATION_STEP)
        if len(rotate_right) > 0:
            for i in range(complete_rotations - len(rotate_right)):
                event_plan = ['RotateLeft'] + event_plan
        elif len(rotate_left) > 0:
            for i in range(complete_rotations - len(rotate_left)):
                event_plan = ['RotateRight'] + event_plan

        return event_plan


# Set the goal position as the position of a state where the object is visible
    def set_goal_obj_state_position(self, goal_object_id):

        # Get goal object information from current agent state
        goal_object_type = goal_object_id.split("_")[0].strip()

        goal_object_states = [s for s in self.knowledge_base.fsm_model.states if f"inspected({goal_object_id})" in s.visible_predicates]

        # DEBUG
        for s in goal_object_states:
            if len([obj for obj in s.visible_objects[goal_object_type] if obj['id'] == goal_object_id]) == 0:
                Logger.write(f"Object {goal_object_id} is not in state {s.id} but is in state visible predicates {s.visible_predicates}")
                exit()

        if len(goal_object_states) == 0:
            Logger.write('WARNING: there are no states where the goal object is within the manipulation '
                         'distance of {} meters.'.format(Configuration.CLOSE_TO_OBJ_DISTANCE))
            return

        # Choose the state which minimizes the goal object distance
        goal_object_states_distances = [[obj['distance'] for obj in s.visible_objects[goal_object_type]
                                         if obj['id'] == goal_object_id][0] for s in goal_object_states]

        feasible_goal = False

        # Get occupancy grid
        grid = self.path_planner.get_occupancy_grid()

        while not feasible_goal and len(goal_object_states) > 0:
            goal_state_index = np.argmin(goal_object_states_distances)
            goal_state = goal_object_states[goal_state_index]

            # Add goal cell marker into occupancy grid
            goal = [goal_state.perceptions['position'][0] * 100, goal_state.perceptions['position'][1] * 100]
            goal_grid = [int(round((goal[0] - self.knowledge_base.map_model.x_min) / self.knowledge_base.map_model.dx)),
                         int(round((goal[1] - self.knowledge_base.map_model.y_min) / self.knowledge_base.map_model.dy))]

            if grid[grid.shape[0] - goal_grid[1]][goal_grid[0]] != 0:
                feasible_goal = True
            else:
                del goal_object_states[goal_state_index]
                del goal_object_states_distances[goal_state_index]

        if not feasible_goal:
            Logger.write('WARNING: There are no feasible states to reach within a distance of 1.5 meters from the goal'
                         'object. This can be due to several reasons: the agent is holding an object which increase its '
                         'encumbrance and collides during the path, or the topview is changed '
                         'or (unlikely) the goal object position has been updated and moved to a not traversable grid cell.')
            return

        # Set event planner goal object position
        goal_pose = [goal_state.perceptions['position'][0], goal_state.perceptions['position'][1],
                     goal_state.perceptions['agent_angle'], int(goal_state.perceptions['camera_tilt'])]  # last number is the camera tilt

        return goal_pose