import random
import re
import numpy as np
import Configuration

np.random.seed(Configuration.RANDOM_SEED)
random.seed(Configuration.RANDOM_SEED)


# Update PDDL state by considering only goal objects.
def update_pddl_state(objects, predicates, objects_counting, objects_scores):

    filtered_objs = [o['id'] for object_type in objects.keys() for o in objects[object_type]
                     if objects_counting[o['id']] >= Configuration.OBJ_COUNT_THRSH
                     and object_type in Configuration.GOAL_OBJECTS]

    filtered_preds = [p for p in predicates if set([o.strip()
                                                    for o in p.split('(')[1].strip()[:-1].split(',')
                                                    if o.strip() != '']).issubset(set(filtered_objs))]

    with open("./PAL/Plan/PDDL/facts.pddl", "r") as f:
        old_pddl_state = [el.strip() for el in f.read().split("\n") if el.strip() != '']
        old_facts = re.findall(":init.*\(:goal", "".join(old_pddl_state))[0]

        old_objs = [el for el in re.findall(":objects(.*?)\)", "++".join(old_pddl_state))[0].split("++") if el.strip() != ""]
        old_state = "\n" + "\n".join(re.findall("\([^()]*\)", old_facts))

        # Replace facts
        new_facts = "\n" + "\n".join(sorted(["({} {})".format(p.strip().split("(")[0],
                                                              " ".join(p.strip().split("(")[1][:-1].split(",")))
                                             for p in filtered_preds]))

        if old_state!="\n":
            new_pddl = "\n".join(old_pddl_state).replace(old_state, new_facts)
        else:
            new_pddl = "\n".join(old_pddl_state).replace("(:init", "(:init" + new_facts)

        # Replace objects
        new_objs = ["{} - {}".format(obj_id, obj_id.split('_')[0]) for obj_id in filtered_objs]

        if len(old_objs) > 0:
            new_pddl = new_pddl.replace("\n".join(old_objs), "\n".join(new_objs))
        else:
            new_pddl = new_pddl.replace("(:objects", "(:objects\n" + "\n".join(new_objs))

    with open("./PAL/Plan/PDDL/facts.pddl", "w") as f:
        f.write(new_pddl)


def get_operator_effects(op_name):

    with open("PAL/Plan/PDDL/domain.pddl", "r") as f:
        data = [el.strip() for el in f.read().split("\n")]

    all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
    action_schema = re.findall(":action {}(.*?)(?::action|$)".format(op_name), all_action_schema)[0]
    effect_neg = re.findall("\(not[^)]*\)\)", action_schema[action_schema.find("effect"):])
    effect_pos = [el for el in re.findall("\([^()]*\)", action_schema[action_schema.find("effect"):])
                     if el not in [el.replace("(not", "").strip()[:-1] for el in effect_neg]
                     and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

    # TMP PATCH
    effect_pos = [e for e in effect_pos if '?x' not in e]
    effect_neg = [e for e in effect_neg if '?x' not in e]

    return effect_pos + effect_neg


def get_goal():
    # Read goal in PDDL problem file
    with open("./PAL/Plan/PDDL/facts.pddl", 'r') as f:
        data = f.read().split("\n")
        for i in range(len(data)):
            if data[i].strip().find("(:goal") != -1:
                return data[i+1]



def set_goal(goal):

    Configuration.GOAL_OBJECTS = [o.split()[-1].strip()[:-1] for o in re.findall("\([^()]*\)", goal) if ' - ' in o]

    # Update goal in PDDL problem file
    with open("./PAL/Plan/PDDL/facts.pddl", 'r') as f:
        data = f.read().split("\n")
        for i in range(len(data)):
            row = data[i]

            if row.strip().find("(:goal") != -1:
                end_index = i + 1

                if data[i].strip().startswith(")"):
                    data[i] = ")\n(:goal \n{} \n))".format(goal)
                else:
                    data[i] = "(:goal \n{} \n))".format(goal)

    with open("./PAL/Plan/PDDL/facts.pddl", 'w') as f:
        [f.write(el + "\n") for el in data[:end_index]]
