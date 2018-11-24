import json
import os
import pandas as pd
import numpy as np


def probas_to_result(probas, unv_int=0.05):
    results = []
    for proba in probas:
        if proba > 0.5 + unv_int:
            results.append(['true', 2 * (proba - 0.5)])
        elif proba < 0.5 - unv_int:
            results.append(['false', 2 * (0.5 - proba)])
        else:
            results.append(['unverified', 0.0])
    return results


def make_submission(subtask, ids, targets_probas, unv_int=0.05):
    results = probas_to_result(targets_probas, unv_int)
    submission = {subtask: {id: result for id, result in zip(ids, results)}}
    with open('submission.json', 'w') as f:
        json.dump(submission, f)
