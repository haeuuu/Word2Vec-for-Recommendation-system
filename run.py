import Playlist2Vec
import AnswerBuilder
from util import *
import pickle
import time
import os

dir = r'C:\Users\haeyu\PycharmProjects\KakaoArena\arena_data'

# data
train, val_que, _ = get_data(dir, test=True)
base_results = load_json(os.path.join(dir, 'results', 'results_gep.json'))

# p2v model
model = Playlist2Vec(train, val_que)
with open(os.path.join(dir, r'model/w2v_128.pkl'), 'rb') as f:
    w2v_model = pickle.load(f)
model.register_w2v(w2v_model)
model.build_p2v()

# answer builder
builder = AnswerBuilder()
builder.register_questions(val_que)
builder.register_answers(val_ans)
builder.register_base_results(base_results)

# build answer
builder.initialize()
for pid, ply in tqdm(builder.val.items()):
    song_rec, tag_rec = model.recommend(pid)
    builder.insert(pid, song_rec, tag_rec)
print(time.time() - s)
print('> only_base_results :', builder.only_base)

# evaluation
res_file_name = os.path.join(dir, r"results/results_w2v.json")
write_json(builder.answers, res_file_name)
evaluator.evaluate(os.path.join(dir, r"answers/val_answers.json"), res_file_name)