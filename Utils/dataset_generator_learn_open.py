import copy

import json
import os

from ai2thor.controller import Controller


if __name__ == "__main__":

    # Initialize empty dataset of episodes
    dataset = []

    # Set scenes
    scenes = []
    for low, high in [(26, 31), (226, 231), (326, 331), (426, 431)]:  # Test scenes
        for i in range(low, high):
            scenes.append('FloorPlan%s' % i)


    # Generate an episode for each scene
    for ep_num, scene in enumerate(scenes):
        controller = Controller(scene=scene, headless=True)
        event = copy.deepcopy(controller.step('Pass'))

        dataset.append({
            "episode": ep_num,
            "scene": scene,
            "goal": "(and (or (and (supervised_opened_book ) (supervised_notopened_book )) (forall (?o1 - book) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_box ) (supervised_notopened_box )) (forall (?o1 - box) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_cabinet ) (supervised_notopened_cabinet )) (forall (?o1 - cabinet) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_drawer ) (supervised_notopened_drawer )) (forall (?o1 - drawer) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_fridge ) (supervised_notopened_fridge )) (forall (?o1 - fridge) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_kettle ) (supervised_notopened_kettle )) (forall (?o1 - kettle) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_laptop ) (supervised_notopened_laptop )) (forall (?o1 - laptop) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_microwave ) (supervised_notopened_microwave )) (forall (?o1 - microwave) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_safe ) (supervised_notopened_safe )) (forall (?o1 - safe) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_showercurtain ) (supervised_notopened_showercurtain )) (forall (?o1 - showercurtain) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_showerdoor ) (supervised_notopened_showerdoor )) (forall (?o1 - showerdoor) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_toilet ) (supervised_notopened_toilet )) (forall (?o1 - toilet) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))))",
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


    with open(os.path.join('../Datasets','test_set_learn_open.json'), 'w') as f:
        json.dump(dataset, f, indent=4)
