import json
import os
import random
import shutil

import numpy as np
import torch.cuda.random

import Configuration
from PAL.Agent import Agent
from Utils import PddlParser, Logger, ResultsPlotter


def main():

    # Set random seed
    np.random.seed(Configuration.RANDOM_SEED)
    random.seed(Configuration.RANDOM_SEED)
    torch.manual_seed(Configuration.RANDOM_SEED)

    # Set environment path variables (for x server communication from WSL2 to Windows GUI)
    if Configuration.USING_WSL2_WINDOWS:
        os.environ['DISPLAY'] = "{}:0.0".format(Configuration.IP_ADDRESS)
        os.environ['LIBGL_ALWAYS_INDIRECT'] = "0"

    # Get episodes dataset
    dataset = json.load(open(os.path.join(Configuration.DATASET_DIR, '{}.json'.format(Configuration.EPISODE_DATASET)), 'r'))

    # Create results log directory
    if os.path.exists(Configuration.RESULTS_DIR):
        Configuration.RESULTS_DIR = f"{Configuration.RESULTS_DIR}({len(os.listdir(Configuration.RESULTS_DIR.split('/')[0]))})"
    os.makedirs(Configuration.RESULTS_DIR, exist_ok=True)

    # Copy PDDL domain file for the learning task
    if Configuration.TASK == Configuration.TASK_LEARN_OPEN:
        shutil.copyfile("PAL/Plan/PDDL/domain_learn_open.pddl", "./PAL/Plan/PDDL/domain.pddl")
    if Configuration.TASK == Configuration.TASK_LEARN_TOGGLE:
        shutil.copyfile("PAL/Plan/PDDL/domain_learn_toggle.pddl", "./PAL/Plan/PDDL/domain.pddl")
    if Configuration.TASK == Configuration.TASK_LEARN_FILL:
        shutil.copyfile("PAL/Plan/PDDL/domain_learn_fill.pddl", "./PAL/Plan/PDDL/domain.pddl")
    if Configuration.TASK == Configuration.TASK_LEARN_DIRTY:
        shutil.copyfile("PAL/Plan/PDDL/domain_learn_dirty.pddl", "./PAL/Plan/PDDL/domain.pddl")

    # Run agent on each episode
    for episode_data in dataset:

        episode = episode_data['episode']
        scene = episode_data['scene']
        goal = episode_data['goal']

        # Create log directories
        Logger.LOG_DIR_PATH = os.path.join(Configuration.RESULTS_DIR, "episode_{}"
                                           .format(len(os.listdir(Configuration.RESULTS_DIR))))
        os.mkdir(Logger.LOG_DIR_PATH)
        Logger.LOG_FILE = open(os.path.join(Logger.LOG_DIR_PATH, "log.txt"), "w")

        # Randomly generate a goal for the scene
        PddlParser.set_goal(goal)

        # Get episode data
        init_position = episode_data['agent_position']
        init_rotation = episode_data['initial_orientation']
        init_horizon = episode_data['initial_horizon']

        if int(episode_data['agent_fov']) != Configuration.FOV:
            Logger.write('Warning: field of view should be {} according to episode data, while default'
                         ' field of view in Configuration.py is {}, I will use default value.'.format(episode_data['agent_fov'], Configuration.FOV))
            # Configuration.FOV = episode_data['agent_fov']

        if Configuration.ROTATION_STEP > Configuration.FOV:
            Logger.write('Warning: agent rotation step ({}) is lower than its field of view ({}). '
                         'Therefore the agent may loop when trying to look at an object which cannot be seen due to'
                         ' blind spots. '.format(Configuration.ROTATION_STEP, Configuration.FOV))

        Logger.write('############# START CONFIGURATION #############\n'
                     'DATASET:{}\n'
                     'EPISODE:{}\n'
                     'SCENE:{}\n'
                     'TASK:{}\n'
                     'RANDOM SEED:{}\n'
                     'GOAL OBJECTS:{}\n'
                     'MAX ITER:{}\n'
                     'VISIBILITY DISTANCE:{}\n'
                     'MOVE STEP:{}\n'
                     'ROTATION DEGREES:{}\n'
                     'FIELD OF VIEW:{}\n'
                     'MAX DISTANCE MANIPULATION (belief):{}\n'
                     'OBJECTS COUNTER THRESHOLD:{}\n'
                     'IoU THRESHOLD:{}\n'
                     'OBJECTS SCORE THRESHOLD:{}\n'
                     'OBJECT DETECTOR GROUND TRUTH:{}\n'
                     'OBJECT DETECTOR:{}\n'
                     '###############################################\n'
                     .format(Configuration.EPISODE_DATASET, episode, scene, Configuration.TASK,
                             Configuration.RANDOM_SEED, Configuration.GOAL_OBJECTS,
                             Configuration.MAX_ITER, Configuration.VISIBILITY_DISTANCE,
                             Configuration.MOVE_STEP, Configuration.ROTATION_STEP, Configuration.FOV,
                             Configuration.MAX_DISTANCE_MANIPULATION, Configuration.OBJ_COUNT_THRSH,
                             Configuration.IOU_THRSH, Configuration.OBJ_SCORE_THRSH,
                             Configuration.GROUND_TRUTH_OBJS, Configuration.OBJ_DETECTOR_PATH))

        # Run agent
        Agent(scene=scene, position=init_position, init_rotation=init_rotation, init_horizon=init_horizon).run()

        # Save results
        ResultsPlotter.plot_self_supervisions(Logger.LOG_DIR_PATH)

        # Copy PDDL state file in result directory
        shutil.copyfile("./PAL/Plan/PDDL/facts.pddl", os.path.join(Logger.LOG_DIR_PATH, "facts_{}.pddl".format(scene)))


if __name__ == "__main__":
    main()
