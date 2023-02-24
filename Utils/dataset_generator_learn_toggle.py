import copy

import json
import os

from ai2thor.controller import Controller

OBJECT_TYPES = ['desklamp', 'candle', 'cellphone', 'faucet', 'laptop', 'showerhead', 'coffeemachine', 'floorlamp', 'desktop', 'microwave', 'toaster', 'television']

if __name__ == "__main__":

    # Initialize empty dataset of episodes
    dataset = []

    # Set scenes
    scenes = []
    # TEST SCENES
    for low, high in [(26, 31), (226, 231), (326, 331), (426, 431)]:
    # VAL SCENES
    # for low, high in [(21, 26), (221, 226), (321, 326), (421, 426)]:
        for i in range(low, high):
            scenes.append('FloorPlan%s' % i)

    all_obj_goals = " ".join([f'(or (and (supervised_toggled_{obj_type}) (supervised_nottoggled_{obj_type}) ) (forall (?o1 - {obj_type}) (and (toggled ?o1) (manipulated ?o1) (scanned ?o1)) ) )'
                               for obj_type in OBJECT_TYPES])
    goal = f"(and {all_obj_goals})"


    # Generate an episode for each scene
    for ep_num, scene in enumerate(scenes):
        print(f'Processing scene {scene} (episode {ep_num})')
        # controller = Controller(scene=scene, headless=True)
        controller = Controller(scene=scene)
        # event = copy.deepcopy(controller.step('Pass'))
        event = controller.step('Pass')

        dataset.append({
            "episode": ep_num,
            "scene": scene,
            "goal": goal,
            "agent_position": {
                "x": event.metadata['agent']['position']['x'],
                "y": event.metadata['agent']['position']['y'],
                "z": event.metadata['agent']['position']['z']
            },
            "agent_rotation": {
                "x": event.metadata['agent']['rotation']['x'],
                "y": event.metadata['agent']['rotation']['y'],
                "z": event.metadata['agent']['rotation']['z']
            },
            "initial_orientation": event.metadata['agent']['rotation']['y'],
            "initial_horizon": event.metadata['agent']['cameraHorizon'],
            "agent_is_standing": event.metadata['agent']['isStanding'],
            "agent_in_high_friction_area": event.metadata['agent']['inHighFrictionArea'],
            "agent_fov": 90.0
        })

        controller.stop()


    with open(os.path.join('../Datasets','test_set_learn_toggle.json'), 'w') as f:
        json.dump(dataset, f, indent=4)