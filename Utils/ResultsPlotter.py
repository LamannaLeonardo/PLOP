import matplotlib.pyplot as plt
import numpy as np

import collections
import os
import pickle

import Configuration
from Utils import Logger


def gen_dataset_open(knowledge_base):
    training_set = []

    # Positive examples
    for obj_type, obj_supervisions in knowledge_base.supervised_objects.items():
        for obj_supervision in obj_supervisions:
            example_label = int('open' in obj_supervision['label'] and 'not_open' not in obj_supervision['label'])
            object_id = obj_supervision['label'].split('(')[1].strip()[:-1]
            training_set.append({'x': obj_supervision['example'], 'y': example_label, 'object_id': object_id})

    positive_example = [e for e in training_set if e['y'] == 1]
    negative_example = [e for e in training_set if e['y'] == 0]

    count_pos = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 1]))

    count_neg = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 0]))

    Logger.write(f"Positive examples: {count_pos}")
    Logger.write(f"Negative examples: {count_neg}")

    print(f'Length of positive examples: {len(positive_example)}\tLength of negative examples: {len(negative_example)}')
    training_set = positive_example + negative_example

    return training_set


def gen_dataset_toggled(knowledge_base):
    training_set = []

    # Positive examples
    for obj_type, obj_supervisions in knowledge_base.supervised_objects.items():
        for obj_supervision in obj_supervisions:
            example_label = int('toggled' in obj_supervision['label'] and 'not_toggled' not in obj_supervision['label'])
            object_id = obj_supervision['label'].split('(')[1].strip()[:-1]

            training_set.append({'x': obj_supervision['example'], 'y': example_label, 'object_id': object_id})

    positive_example = [e for e in training_set if e['y'] == 1]
    negative_example = [e for e in training_set if e['y'] == 0]

    count_pos = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 1]))

    count_neg = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 0]))

    Logger.write(f"Positive examples: {count_pos}")
    Logger.write(f"Negative examples: {count_neg}")

    print(f'Length of positive examples: {len(positive_example)}\tLength of negative examples: {len(negative_example)}')
    training_set = positive_example + negative_example

    return training_set


def gen_dataset_filled(knowledge_base):
    training_set = []

    # Positive examples
    for obj_type, obj_supervisions in knowledge_base.supervised_objects.items():
        for obj_supervision in obj_supervisions:
            example_label = int('filled' in obj_supervision['label'] and 'not_filled' not in obj_supervision['label'])
            object_id = obj_supervision['label'].split('(')[1].strip()[:-1]
            training_set.append({'x': obj_supervision['example'], 'y': example_label, 'object_id': object_id})

    positive_example = [e for e in training_set if e['y'] == 1]
    negative_example = [e for e in training_set if e['y'] == 0]

    count_pos = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 1]))

    count_neg = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 0]))

    Logger.write(f"Positive examples: {count_pos}")
    Logger.write(f"Negative examples: {count_neg}")

    print(f'Length of positive examples: {len(positive_example)}\tLength of negative examples: {len(negative_example)}')
    training_set = positive_example + negative_example

    return training_set


def gen_dataset_dirty(knowledge_base):
    training_set = []

    # Positive examples
    for obj_type, obj_supervisions in knowledge_base.supervised_objects.items():
        for obj_supervision in obj_supervisions:
            example_label = int('dirty' in obj_supervision['label'] and 'not_dirty' not in obj_supervision['label'])
            object_id = obj_supervision['label'].split('(')[1].strip()[:-1]
            training_set.append({'x': obj_supervision['example'], 'y': example_label, 'object_id': object_id})

    positive_example = [e for e in training_set if e['y'] == 1]
    negative_example = [e for e in training_set if e['y'] == 0]

    count_pos = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 1]))

    count_neg = dict(collections.Counter([e['object_id'].split('_')[0] for e in training_set if e['y'] == 0]))

    Logger.write(f"Positive examples: {count_pos}")
    Logger.write(f"Negative examples: {count_neg}")

    print(f'Length of positive examples: {len(positive_example)}\tLength of negative examples: {len(negative_example)}')
    training_set = positive_example + negative_example

    return training_set


def save_collected_data(knowledge_base, scene):
    if Configuration.TASK == Configuration.TASK_LEARN_OPEN:
        training_set = gen_dataset_open(knowledge_base)
        pickle.dump(training_set, open(os.path.join(Logger.LOG_DIR_PATH, f'open_train_{scene}.pkl'), 'wb'))
    elif Configuration.TASK == Configuration.TASK_LEARN_TOGGLE:
        training_set = gen_dataset_toggled(knowledge_base)
        pickle.dump(training_set, open(os.path.join(Logger.LOG_DIR_PATH, f'toggle_train_{scene}.pkl'), 'wb'))
    elif Configuration.TASK == Configuration.TASK_LEARN_FILL:
        training_set = gen_dataset_filled(knowledge_base)
        pickle.dump(training_set, open(os.path.join(Logger.LOG_DIR_PATH, f'fill_train_{scene}.pkl'), 'wb'))
    elif Configuration.TASK == Configuration.TASK_LEARN_DIRTY:
        training_set = gen_dataset_dirty(knowledge_base)
        pickle.dump(training_set, open(os.path.join(Logger.LOG_DIR_PATH, f'dirty_train_{scene}.pkl'), 'wb'))





def plotCurve(X, Y, xlabel, ylabel, title, file_name):
    nticks=5
    plt.plot(X, Y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(X)
    plt.xticks(np.arange(min(X), max(X) + 1, int(max(X)/nticks)))
    plt.title(title)
    plt.savefig(file_name)
    plt.close()


def plotBar(names, values, xlabel, ylabel, title, file_name):
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()


def plot_self_supervisions(episode_dir):

    plt.style.use('ggplot')
    with open(os.path.join(episode_dir, 'log.txt'), 'r') as f:
        steps = [r for r in f.read().split('\n') if r.split(':')[0].isdigit()]

    X = list(range(len(steps)))
    Y = [r[:r.index('self-supervised')].split()[-1] for r in steps]

    plt.tight_layout()

    plotCurve(X, Y, 'Steps', 'Self-supervised examples', 'Self-supervision', os.path.join(episode_dir, 'self_supervision.png'))


