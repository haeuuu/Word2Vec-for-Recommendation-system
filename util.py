import json
import os

def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj

def get_data(dir, test = True):
    if test:
        train_path = os.path.join(dir, 'org/train.json')
        val_path = os.path.join(dir, 'qustions/val_questions.json')
    else:
        pass

    base_path = os.path.join(dir, 'base_results_gep.json')

    train = load_json(train_path)
    val = load_json(val_path)
    base_res = load_json(base_path)

    return train, val, base_res
