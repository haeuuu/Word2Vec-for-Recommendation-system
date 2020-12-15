import json
import os, io
import numpy as np

def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    with io.open(fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def get_data(dir = 'arena_data', test = True):
    if test:
        train_path = os.path.join(dir, 'orig' ,'train.json')
        val_que_path = os.path.join(dir, 'questions','val_questions.json')
        val_ans_path = os.path.join(dir, 'answers','val_answers.json')
    else:
        train_path = os.path.join(dir, 'submission' ,'train.json')
        val_path = os.path.join(dir, 'submission','val_answers.json')

    train = load_json(train_path)
    val_que = load_json(val_que_path)
    val_ans = load_json(val_ans_path)

    return train, val_que, val_ans
