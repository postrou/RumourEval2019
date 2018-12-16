import json
import os
import sys; sys.path.insert(0, '../organizers_baseline/preprocessing')

import numpy as np
import pandas as pd

from tree2branches import tree2branches
from features import *


def load_task_targets(data_dir: str):
    targets = {}
    for event_dir in next(os.walk(data_dir))[1]:
        for veracity in next(os.walk(os.path.join(data_dir, event_dir)))[1]:
            for tweet_id in next(os.walk(os.path.join(data_dir, event_dir, veracity)))[1]:
                targets.update({tweet_id: 0 if veracity == 'non-rumours' else 1})
        # non-rumours
        # for tweet in next(os.walk(os.path.join(data_dir, 'non-rumours', event_dir)))[1]:
            # targets.update({tweet: 'false'})

    return pd.Series(targets)

def load_rumours_data(data_dir):
    rumours_source_dict = {}

    for event_name in next(os.walk(data_dir))[1]:
        # print(next(os.walk(data_dir))[1])
        for veracity in next(os.walk(os.path.join(data_dir, event_name)))[1]:
            # print
            for tweet_id in next(os.walk(os.path.join(data_dir, event_name, veracity)))[1]:
                tweet_path = os.path.join(data_dir, event_name, veracity, tweet_id, 'source-tweet')
                with open(os.path.join(tweet_path, tweet_id + '.json'), 'r') as f:
                    rumour = json.load(f)
                    rumour_data = handle_rumour(rumour)
                rumours_source_dict.update(rumour_data)

    return rumours_source_dict


def handle_rumour(rumour):
    rumour_data = {str(rumour['id']):{'text': text_preprocess(rumour['text'])}}
    return rumour_data


def build_dataset(data_dir):
    rumours_source_dict = load_rumours_data(data_dir)

    b_y = load_task_targets(data_dir)

    b_data = pd.DataFrame.from_dict(rumours_source_dict, orient='index')

    b_data = b_data.assign(veracity=b_y)

    return b_data

#--------------------------LINEAR------------------------------------

def lin_rumour_data(a_data, b_data, struct):
    X_dict = {}
    branches = tree2branches(struct)

    n_support = 0
    n_deny = 0
    n = 0

    for i, branch in enumerate(branches):
        n += len(branch[1:])

        for reply in branch[1:]:
            try:
                sqdc = a_data.loc[reply, 'sqdc']
            except KeyError:
                continue

            if sqdc == 'support':
                n_support += 1
            elif sqdc == 'deny':
                n_deny += 1

        try:
            veracity = b_data.loc[branch[0], 'veracity']
            X_dict.update({branch[0]: {'support': n_support / n, 'deny': n_deny / n, 'veracity': veracity}})
        except KeyError:
            pass

        if i < len(branches) - 1:
            if branches[i + 1][0] != branch[0]:
                n_support = 0
                n_deny = 0
                n = 0
    data = pd.DataFrame.from_dict(X_dict, orient='index')

    return data


def remove_unverified(data):
    data = data.loc[data.loc[:, 'veracity'].isin(['true', 'false'])]
    data.loc[:, 'veracity'] = [1 if v == 'true' else 0 for v in data.loc[:, 'veracity']]

    return data.loc[:, ['support', 'deny']], data.loc[:, 'veracity']
