import collections
import copy

import numpy as np

import Configuration
from Utils import Logger

import os


class PathPlanner:

    def __init__(self, knowledge_base):

        self.knowledge_base = knowledge_base

        # First goal position in the map corner so that agent has to entirely explore the environment
        # before changing goal position
        self.goal_position = Configuration.INIT_GOAL_POS


    def path_planning(self):

        # Get occupancy grid
        grid = copy.deepcopy(self.get_occupancy_grid())

        # Add agent starting position into occupancy grid
        agent_position = self.knowledge_base.current_state.perceptions['position']
        # start = [agent_position['x']*100, agent_position['y']*100]
        start = [agent_position[0]*100, agent_position[1]*100]
        start_grid = (int(round((start[0]-self.knowledge_base.map_model.x_min)/self.knowledge_base.map_model.dx)),
                      int(round((start[1]-self.knowledge_base.map_model.y_min)/self.knowledge_base.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

        # Add goal cell marker into occupancy grid
        goal = [self.goal_position[0]*100, self.goal_position[1]*100]
        goal_grid = [int(round((goal[0]-self.knowledge_base.map_model.x_min)/self.knowledge_base.map_model.dx)),
                     int(round((goal[1]-self.knowledge_base.map_model.y_min)/self.knowledge_base.map_model.dy))]

        # Check if goal cell is traversible or not
        if grid[grid.shape[0] - goal_grid[1]][goal_grid[0]] == 0:
            return None

        grid[grid.shape[0] - goal_grid[1]][goal_grid[0]] = 2  # 2 is the goal integer identifier in the grid

        # DEBUG plot grid
        if Configuration.PRINT_TOP_VIEW_GRID_PLAN_IMAGES:
            grid_debug = copy.deepcopy(grid)
            grid_debug[(grid_debug==1)] = 255
            grid_debug[(grid_debug==2)] = 180
            grid_debug[start_grid[1]][start_grid[0]] = 100
            Logger.save_img("topview_grid_noplan.png", grid_debug)

        # Compute plan into resized occupancy grid
        grid_plan = self.bfs(grid, start_grid, goal)
        plan = self.compile_plan(grid_plan)

        # Plot grid with plan for debugging
        if Configuration.PRINT_TOP_VIEW_GRID_PLAN_IMAGES:
            if grid_plan is not None:
                idx_i, idx_j = zip(*grid_plan)
                grid_debug[idx_j, idx_i] = 220 # draw plan
                grid_debug[(grid_debug==2)] = 180 # draw goal position
                grid_debug[start_grid[1]][start_grid[0]] = 100 # draw agent position
                count = len([el for el in os.listdir(Logger.LOG_DIR_PATH) if 'grid' in el and 'noplan' not in el])
                Logger.save_img("topview_grid_{}.png".format(count), grid_debug)

        return plan


    # Add more goal points around the goal point and within the manipulation distance
    def path_planning_greedy(self, start_position, goal_position, around_goal_distance, non_goal_grid_cells=None):

        # Get occupancy grid
        grid = copy.deepcopy(self.get_occupancy_grid())
        single_goal_grid = copy.deepcopy(grid)

        # Add agent starting position into occupancy grid
        start = [start_position['x'] * 100, start_position['y'] * 100]
        start_grid = (int(round((start[0] - self.knowledge_base.map_model.x_min) / self.knowledge_base.map_model.dx)),
                      int(round((start[1] - self.knowledge_base.map_model.y_min) / self.knowledge_base.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

        # Add goal cell marker into occupancy grid
        goal = [goal_position[0] * 100, goal_position[1] * 100]
        goal_grid = [int(round((goal[0] - self.knowledge_base.map_model.x_min) / self.knowledge_base.map_model.dx)),
                     int(round((goal[1] - self.knowledge_base.map_model.y_min) / self.knowledge_base.map_model.dy))]

        # Add greedy goal cells marker into occupancy grid, i.e. positions within manipulation distance from goal point
        for i in range(- int(Configuration.MAX_DISTANCE_MANIPULATION / (Configuration.MOVE_STEP*100)),
                       int(Configuration.MAX_DISTANCE_MANIPULATION / (Configuration.MOVE_STEP*100)) + 1):
            for j in range(- int(Configuration.MAX_DISTANCE_MANIPULATION / (Configuration.MOVE_STEP*100)),
                           int(Configuration.MAX_DISTANCE_MANIPULATION / (Configuration.MOVE_STEP*100)) + 1):
                if grid[grid.shape[0] - (goal_grid[1] + i)][goal_grid[0] + j] != 0:
                    if non_goal_grid_cells is None \
                            or (grid.shape[0] - (goal_grid[1] + i), goal_grid[0] + j) not in non_goal_grid_cells:

                        if grid[grid.shape[0] - (goal_grid[1] + i)][goal_grid[0] + j] != 0:

                            # Check if the distance between cell center and goal object is lower than the MAX_DISTANCE_MANIPULATION
                            cell_pos = [goal_position[0] + i*Configuration.MOVE_STEP, goal_position[1] + j*Configuration.MOVE_STEP, Configuration.CAMERA_HEIGHT]

                            if np.linalg.norm(np.array(goal_position) - np.array(cell_pos)) < around_goal_distance:

                                single_goal_grid[grid.shape[0] - (goal_grid[1] + i)][goal_grid[0] + j] = 2

                                # modo 1: controllo feasibility
                                # single_goal_path = self.bfs(single_goal_grid, start_grid, goal)
                                # if single_goal_path is not None:
                                #     grid[grid.shape[0] - (goal_grid[1] + i)][goal_grid[0] + j] = 2   # 2 is the goal integer identifier in the grid
                                #
                                # single_goal_grid[grid.shape[0] - (goal_grid[1] + i)][goal_grid[0] + j] = 1

                                # modo 2: assumo feasibility e gestisco piano nullo
                                grid[grid.shape[0] - (goal_grid[1] + i)][goal_grid[0] + j] = 2

        # DEBUG plot grid
        if Configuration.PRINT_TOP_VIEW_GRID_PLAN_IMAGES:
            grid_debug = copy.deepcopy(grid)
            grid_debug[(grid_debug == 1)] = 255
            grid_debug[(grid_debug == 2)] = 180
            grid_debug[start_grid[1]][start_grid[0]] = 100
            Logger.save_img("topview_grid_noplan.png", grid_debug)

        # Check if agent position is already a goal one
        if grid[start_grid[1]][start_grid[0]] == 2:
            return []

        # Compute plan into resized occupancy grid
        grid_plan = self.bfs(grid, start_grid, goal)
        plan = self.compile_plan(grid_plan)

        # Plot grid with plan for debugging
        if Configuration.PRINT_TOP_VIEW_GRID_PLAN_IMAGES:
            if grid_plan is not None:
                idx_i, idx_j = zip(*grid_plan)
                grid_debug[idx_j, idx_i] = 220  # draw plan
                grid_debug[(grid_debug == 2)] = 180  # draw goal position
                grid_debug[start_grid[1]][start_grid[0]] = 100  # draw agent position
                count = len([el for el in os.listdir(Logger.LOG_DIR_PATH) if 'grid' in el and 'noplan' not in el])
                Logger.save_img("topview_grid_{}.png".format(count), grid_debug)

        return plan


    def get_occupancy_grid(self):
        # Add collision cells into occupancy grid
        for cell in self.knowledge_base.map_model.collision_cells:
            self.knowledge_base.map_model.grid[self.knowledge_base.map_model.grid.shape[0] - cell[0]][cell[1]] = 0
        return self.knowledge_base.map_model.grid


    def bfs(self, grid, start, goal):
        """
        Example call
        wall, clear, goal = "#", ".", "*"
        width, height = 10, 5
        grid = ["..........",
                "..*#...##.",
                "..##...#*.",
                ".....###..",
                "......*..."]
        path = bfs(grid, (5, 2))
        :param grid: occupancy map
        :param start: starting grid cell
        :return:
        """
        wall = 0
        goal = 2
        height = grid.shape[0]
        width = grid.shape[1]
        queue = collections.deque([[start]])
        seen = set([tuple(start)])
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if grid[y][x] == goal:
                return path
            adjacent_cells = ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
            if Configuration.DIAGONAL_MOVE:
                adjacent_cells = ((x+1, y), (x-1, y), (x, y+1), (x, y-1), (x-1, y-1), (x-1, y+1), (x+1, y+1), (x+1, y-1))

            for x2, y2 in adjacent_cells:
                if 0 <= x2 < width and 0 <= y2 < height \
                        and grid[y2][x2] != wall \
                        and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))


    def compile_plan(self, plan):

        # If no plan can be computed, return none action list
        if plan is None:
            return None

        plan_actions = []

        dtheta = Configuration.ROTATION_STEP  # dtheta degrees of agent rotation action
        agent_theta = copy.deepcopy(self.knowledge_base.current_state.perceptions['agent_angle'])

        for i in range(len(plan) - 1):
            relative_angle = None

            # Get relative angle between two subsequent grid cells
            # plan is a list of tuples (column, row)
            if plan[i+1][1] == plan[i][1] - 1 and plan[i+1][0] == plan[i][0]:
                relative_angle = 90
            elif plan[i + 1][1] == plan[i][1] and plan[i + 1][0] == plan[i][0] - 1:
                relative_angle = 180
            elif plan[i + 1][1] == plan[i][1] + 1 and plan[i + 1][0] == plan[i][0]:
                relative_angle = 270
            elif plan[i + 1][1] == plan[i][1] and plan[i + 1][0] == plan[i][0] + 1:
                relative_angle = 0
            elif plan[i + 1][1] == plan[i][1] - 1 and plan[i + 1][0] == plan[i][0] + 1:
                relative_angle = 45
            elif plan[i + 1][1] == plan[i][1] - 1 and plan[i + 1][0] == plan[i][0] - 1:
                relative_angle = 135
            elif plan[i + 1][1] == plan[i][1] + 1 and plan[i + 1][0] == plan[i][0] - 1:
                relative_angle = 225
            elif plan[i + 1][1] == plan[i][1] + 1 and plan[i + 1][0] == plan[i][0] + 1:
                relative_angle = 315

            move_angle = round(relative_angle - agent_theta)

            # Optimize rotations, i.e. instead of 3 rotating left actions of 90 degrees => 1 rotating right action of 90
            if move_angle > 180:
                move_angle = move_angle - 360
            elif move_angle < -180:
                move_angle = 360 + move_angle

            # Add rotation actions
            if move_angle > 0:
                for _ in range(abs(move_angle // dtheta)):
                    plan_actions.append('RotateLeft')
                    agent_theta += dtheta
                    agent_theta = agent_theta % 360
            else:
                for _ in range(abs(move_angle // -dtheta)):
                    plan_actions.append('RotateRight')
                    agent_theta -= dtheta
                    agent_theta = agent_theta % 360

            # Add move forward action
            plan_actions.append('MoveAhead')

        return plan_actions
