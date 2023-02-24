import copy
import random

import Configuration
from PAL.Act.ActionCompiler import ActionCompiler
from PAL.Plan.PDDLPlanner import PDDLPlanner
from Utils import Logger


class EventPlanner:

    def __init__(self, knowledge_base):
        random.seed(0)

        # Set knowledge base
        self.knowledge_base = knowledge_base

        # Set pddl planner
        self.pddl_planner = PDDLPlanner()

        # Set symbolic action compiler (into low level actions)
        self.action_compiler = ActionCompiler(self.knowledge_base)

        # Set path plan
        self.path_plan = None

        # Set event plan
        self.event_plan = None

        # Current event planner subgoal
        self.subgoal = None


    def explore(self):

        # Adjust agent camera inclination
        if int(self.knowledge_base.current_state.perceptions['camera_tilt']) < 0:
            return 'LookUp'
        elif int(self.knowledge_base.current_state.perceptions['camera_tilt']) > 0:
            return 'LookDown'

        # Replan if no previously computed path plan is available
        if self.path_plan is None:
            self.path_plan = self.action_compiler.path_planner.path_planning()

        # Adjust plan whether the agent has collided while rotating and holding an object
        if self.action_compiler.rotation_collision and self.path_plan is not None:
            self.adjust_path_plan_rotations()
            self.action_compiler.rotation_collision = False

        # If goal position is either unreachable or has been reached, sample a new goal position and explore
        while self.path_plan is None or len(self.path_plan) == 0:

            # The first goal position is very far from the agent, i.e., outside of the real map of the environment,
            # therefore when no path_plan can be computed to reach such a position then the agent has entirely explored
            # the environment (and the predicate explored() becomes true in the symbolic state)
            if 'explored()' not in self.knowledge_base.all_predicates:
                self.knowledge_base.add_predicate('explored()')

            # Sample a new goal position
            self.action_compiler.path_planner.goal_position = self.sample_goal_position()

            # DEBUG
            print("[Debug] New goal position is: {}".format(self.action_compiler.path_planner.goal_position))

            # Compute a path plan towards the new sampled goal position
            self.path_plan = self.action_compiler.path_planner.path_planning()

            # Check if the agent got stuck in a cramped area
            if self.agent_is_blocked():
                Logger.write('Warning: agent is blocked, clearing the area around the agent.')
                self.free_agent_area()

        return self.path_plan.pop(0)


    def sample_goal_position(self):

        # while self.path_plan is None:
        agent_x_pos = self.knowledge_base.current_state.perceptions['position'][0]
        agent_y_pos = self.knowledge_base.current_state.perceptions['position'][1]
        sampling_distance = 150  # centimeters

        sampled_x_pos = random.randint(max(self.knowledge_base.map_model.x_min + (Configuration.MOVE_STEP*100*2),
                                                                                  int(agent_x_pos*100) - sampling_distance),
                                                                              min(self.knowledge_base.map_model.x_max - (Configuration.MOVE_STEP*100*2),
                                                                                  int(agent_x_pos*100) + sampling_distance)) / 100
        sampled_y_pos = random.randint(max(self.knowledge_base.map_model.y_min + (Configuration.MOVE_STEP * 100 * 2),
                                                                                  int(agent_y_pos * 100) - sampling_distance),
                                                                              min(self.knowledge_base.map_model.y_max - (Configuration.MOVE_STEP * 100 * 2),
                                                                                  int(agent_y_pos * 100) + sampling_distance)) / 100
        sampled_pos = [sampled_x_pos, sampled_y_pos]

        return sampled_pos


    # Set the object position as a goal one
    def agent_is_blocked(self):

        agent_position = self.knowledge_base.current_state.perceptions['position']

        # Get occupancy grid
        grid = copy.deepcopy(self.action_compiler.path_planner.get_occupancy_grid())

        # Add agent starting position into occupancy grid
        start = [agent_position[0]*100, agent_position[1]*100]
        start_grid = (int(round((start[0]-self.knowledge_base.map_model.x_min)/self.knowledge_base.map_model.dx)),
                      int(round((start[1]-self.knowledge_base.map_model.y_min)/self.knowledge_base.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

        if not Configuration.DIAGONAL_MOVE:
            agent_blocked = grid[start_grid[1] + 1, start_grid[0]] == 0 \
                            and grid[start_grid[1] - 1, start_grid[0]] == 0 \
                            and grid[start_grid[1], start_grid[0] + 1] == 0 \
                            and grid[start_grid[1], start_grid[0] - 1] == 0
        else:
            agent_blocked = grid[start_grid[1] + 1, start_grid[0]] == 0 \
                            and grid[start_grid[1] - 1, start_grid[0]] == 0 \
                            and grid[start_grid[1], start_grid[0] + 1] == 0 \
                            and grid[start_grid[1], start_grid[0] - 1] == 0\
                            and grid[start_grid[1] - 1, start_grid[0] - 1] == 0\
                            and grid[start_grid[1] - 1, start_grid[0] + 1] == 0\
                            and grid[start_grid[1] + 1, start_grid[0] - 1] == 0\
                            and grid[start_grid[1] + 1, start_grid[0] + 1] == 0

        return agent_blocked


    # Set the object position as a goal one
    def free_agent_area(self):

        agent_position = self.knowledge_base.current_state.perceptions['position']

        # Get occupancy grid
        grid = self.action_compiler.path_planner.get_occupancy_grid()

        # Add agent starting position into occupancy grid
        start = [agent_position[0]*100, agent_position[1]*100]
        start_grid = (int(round((start[0]-self.knowledge_base.map_model.x_min)/self.knowledge_base.map_model.dx)),
                      int(round((start[1]-self.knowledge_base.map_model.y_min)/self.knowledge_base.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

        if not Configuration.DIAGONAL_MOVE:
            agent_area_cells = [(start_grid[1] + 1, start_grid[0]),
                                (start_grid[1], start_grid[0] + 1),
                                (start_grid[1] - 1, start_grid[0]),
                                (start_grid[1], start_grid[0] - 1)]
        else:
            agent_area_cells = [(start_grid[1] + 1, start_grid[0]),
                                (start_grid[1], start_grid[0] + 1),
                                (start_grid[1] - 1, start_grid[0]),
                                (start_grid[1], start_grid[0] - 1),
                                (start_grid[1] + 1, start_grid[0] - 1),
                                (start_grid[1] - 1, start_grid[0] - 1),
                                (start_grid[1] + 1, start_grid[0] + 1),
                                (start_grid[1] - 1, start_grid[0] + 1)]

        for cell in agent_area_cells:
            self.knowledge_base.map_model.grid[cell[0], cell[1]] = 1

            if [self.knowledge_base.map_model.grid.shape[0] - cell[0], cell[1]] in self.knowledge_base.map_model.collision_cells:
                self.knowledge_base.map_model.collision_cells.remove([self.knowledge_base.map_model.grid.shape[0] - cell[0], cell[1]])


    def event_planning(self):

        fsm_model = self.knowledge_base.fsm_model

        if self.subgoal.split("(")[0].strip().lower().startswith('get_close_and_look_at'):

            self.event_plan = self.action_compiler.compile(self.subgoal)

            if self.event_plan is None:
                unreachable_obj_id = self.subgoal.split('(')[1][:-1].strip().lower()

                unreachable_goal_states = [s for s in self.knowledge_base.fsm_model.states
                                           if f"inspected({unreachable_obj_id})" in s.visible_predicates]
                # Remove inspected predicate from all states close to the current one and from knowledge base
                self.knowledge_base.fsm_model.states = [s for s in self.knowledge_base.fsm_model.states
                                                        if s not in unreachable_goal_states]
                self.knowledge_base.remove_predicate(f'inspected({unreachable_obj_id})')
                self.knowledge_base.update_pddl_state()
                return None

            if len(self.event_plan) == 0:
                # DEBUG
                # print('[Debug] Goal object is in agent view.')
                self.subgoal = None
                self.event_plan = None
            else:
                return self.event_plan.pop(0)


        elif self.subgoal.split("(")[0].strip() == "INSPECT":

            # Compile the symbolic action into low-level ones
            if self.event_plan is None or len(self.event_plan) == 1:
                self.event_plan = self.action_compiler.compile(self.subgoal)

            # Check if goal object inspection is feasible
            if self.event_plan is None:
                self.action_compiler.explored_goal_cells = list(range(Configuration.MAX_EXPLORED_GOAL_CELLS_INSPECT))
                return self.explore()

            # If all compiled actions (but a fictitious one) have been executed, check if the inspection succeeded,
            # i.e., if there is a state where the object has been detected and its distance from the agent is lower
            # than the manipulation one
            if len(self.event_plan) == 1:

                # Check if goal object has been seen in a state where its distance is lower than the manipulation one
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                goal_object_type = goal_object_id.split("_")[0].strip()
                goal_object_states = [s for s in fsm_model.states
                                      if goal_object_type in s.visible_objects.keys()
                                      and goal_object_id in [obj['id'] for obj in s.visible_objects[goal_object_type]
                                                             if obj['distance'] < Configuration.CLOSE_TO_OBJ_DISTANCE]]
                if len(goal_object_states) == 0:
                    self.event_plan = None

                    # Get agent position
                    start = [self.knowledge_base.current_state.perceptions['position'][0] * 100,
                             self.knowledge_base.current_state.perceptions['position'][1] * 100]
                    start_grid = (int(round((start[0] - self.knowledge_base.map_model.x_min) / self.knowledge_base.map_model.dx)),
                                  int(round((start[1] - self.knowledge_base.map_model.y_min) / self.knowledge_base.map_model.dy)))
                    grid = self.action_compiler.path_planner.get_occupancy_grid()
                    start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

                    self.action_compiler.explored_goal_cells.append((start_grid[1], start_grid[0]))

                    # DEBUG
                    Logger.write('Adding a useless goal cell: {}'.format(len(self.action_compiler.explored_goal_cells)))
                    self.event_plan = self.action_compiler.compile(self.subgoal)

                    # Check if goal object inspection is feasible
                    if self.event_plan is None:
                        self.action_compiler.explored_goal_cells = list(range(Configuration.MAX_EXPLORED_GOAL_CELLS_INSPECT))
                        return self.explore()

                    return self.event_plan.pop(0)

                else:
                    # DEBUG
                    # print('[Debug] Goal object has been successfully inspected.')
                    self.subgoal = None
                    self.event_plan = None
                    self.action_compiler.explored_goal_cells = []
            else:
                return self.event_plan.pop(0)


        elif self.subgoal.split("(")[0].startswith("SCAN"):

            # Compile the symbolic action into low-level ones
            self.event_plan = self.action_compiler.compile(self.subgoal)

            if self.event_plan is not None and len(self.event_plan) == 1:
                # Get agent position
                start = [self.knowledge_base.current_state.perceptions['position'][0] * 100,
                         self.knowledge_base.current_state.perceptions['position'][1] * 100]
                start_grid = (int(round((start[0] - self.knowledge_base.map_model.x_min) / self.knowledge_base.map_model.dx)),
                              int(round((start[1] - self.knowledge_base.map_model.y_min) / self.knowledge_base.map_model.dy)))
                grid = self.action_compiler.path_planner.get_occupancy_grid()
                start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

                # Add explored goal cell
                self.action_compiler.explored_goal_cells.append((start_grid[1], start_grid[0]))

                self.event_plan = self.action_compiler.compile(self.subgoal)

            # Check if goal object inspection is feasible
            if self.event_plan is None:
                # DEBUG
                print('Goal object cannot be scanned from {} positions because there are not enough reachable ones.'.format(Configuration.MAX_EXPLORED_GOAL_CELLS_SCAN))

                self.action_compiler.explored_goal_cells = []
                self.event_plan = [self.explore()]
                return self.event_plan.pop(0)
            else:
                return self.event_plan.pop(0)

        elif self.subgoal.split("(")[0].strip() in ["OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF", "FILL", "UNFILL",
                                                    "DIRTY", "CLEAN"]:

            if self.event_plan is None:
                self.event_plan = self.action_compiler.compile(self.subgoal)

            # If all low-level actions have been executed
            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
            # Return the next low-level action in the compiled sequence
            else:
                return self.event_plan.pop(0)

        elif self.subgoal.split("(")[0].strip() in ["PICKUP", "PUTON"]:

            self.event_plan = self.action_compiler.compile(self.subgoal)

            # If all low-level actions have been executed
            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
            # Return the next low-level action in the compiled sequence
            else:
                return self.event_plan.pop(0)

        elif self.subgoal.split("(")[0].strip() == "STOP":

            if self.event_plan is None:
                self.event_plan = ["Stop"]

            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
            else:
                return self.event_plan.pop(0)


# Adjust plan whether the agent has collided while rotating and holding an object
    def adjust_path_plan_rotations(self):

            rotate_left = []
            rotate_right = []
            for action in self.path_plan:
                if action == 'RotateLeft':
                    rotate_left.append('RotateLeft')
                elif action == 'RotateRight':
                    rotate_right.append('RotateRight')
                else:
                    break

            self.path_plan = self.path_plan[len(rotate_left) + len(rotate_right):]

            assert len(rotate_right) == 0 or len(rotate_left) == 0, 'Check adjust_path_plan_rotations() in EventPlanner.py'

            complete_rotations = int(360 / Configuration.ROTATION_STEP)
            if len(rotate_right) > 0:
                for i in range(complete_rotations - len(rotate_right)):
                    self.path_plan = ['RotateLeft'] + self.path_plan
            elif len(rotate_left) > 0:
                for i in range(complete_rotations - len(rotate_left)):
                    self.path_plan = ['RotateRight'] + self.path_plan
