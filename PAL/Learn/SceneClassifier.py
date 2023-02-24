import pickle
from collections import defaultdict

import Configuration


class SceneClassifier:

    def __init__(self):
        self.scene_objects = None

    def get_visible_predicates(self, visible_objects, rgb_img):

        visible_predicates = defaultdict(dict)

        for obj_type in visible_objects:
            for obj in visible_objects[obj_type]:
                obj_name = obj['id']
                obj_type = obj_name.split("_")[0]
                obj_discovered = True
                obj_close = bool(obj['distance'] <= Configuration.CLOSE_TO_OBJ_DISTANCE)

                obj_visible = True

                obj_inspected = obj_close and obj_visible

                # Crop image to object bbox
                obj_bbox = [int(coord) for coord in obj['bb']['corner_points']]
                # obj_img_rgb = rgb_img[
                #               max(0, obj_bbox[1] - 3): min(rgb_img.shape[0], obj_bbox[3] + 4),
                #               max(0, obj_bbox[0] - 3): min(rgb_img.shape[1], obj_bbox[2] + 4),
                #               :
                #               ]

                obj_openable = obj_type in Configuration.OPENABLE_OBJS
                obj_dirtyable = obj_type in Configuration.DIRTYABLE_OBJS
                obj_toggleable = obj_type in Configuration.TOGGLEABLE_OBJS
                obj_fillable = obj_type in Configuration.FILLABLE_OBJS
                obj_pickable = obj_type in Configuration.PICKUPABLE_OBJS
                obj_receptacle = obj_type in Configuration.RECEPTACLE_OBJS

                visible_predicates[obj_name] = {
                                                'discovered': obj_discovered,
                                                'viewing': obj_visible,
                                                'close_to': obj_close,
                                                'inspected': obj_inspected,
                                                'openable': obj_openable,
                                                'dirtyable': obj_dirtyable,
                                                'toggleable': obj_toggleable,
                                                'fillable': obj_fillable,
                                                'pickupable': obj_pickable,
                                                'receptacle': obj_receptacle}

        # Generate visible predicates list
        visible_predicates = dict(visible_predicates)

        # Get unary predicates
        visible_predicates_list = ["{}({})".format(k2, k) for k, v in visible_predicates.items()
                                   for k2, v2 in v.items() if type(v2) == type(True) and v2]

        return visible_predicates_list
