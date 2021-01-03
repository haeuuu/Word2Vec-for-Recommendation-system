import Playlist2Vec
import AnswerBuilder
from util import *
import pickle
import time
import os

default_dir = r'C:\Users\haeyu\PycharmProjects\KakaoArena\arena_data'

# data
train_path = os.path.join(default_dir, r'orig/train.json')
val_que_path = os.path.join(default_dir, r'questions/val_questions.json')
val_ans_path = os.path.join(default_dir, r'answers/val_answers.json')
results_gep_path = os.path.join(default_dir, 'results', 'results_gep.json')

# p2v model
model = Playlist2Vec(train_path, val_que_path)
with open(os.path.join(default_dir, r'model/w2v_128.pkl'), 'rb') as f:
    w2v_model = pickle.load(f)
model.register_w2v(w2v_model)
model.build_p2v(normalize_song = False, normalize_tag = True, normalize_title = True ,
                song_weight = 1, tag_weight = 1 ,use_bm25 = True)

# answer builder
builder = AnswerBuilder()
builder.register_questions(val_que_path)
builder.register_answers(val_ans_path)
builder.register_base_results(results_gep_path)

# build answer
s = time.time()
builder.initialize()
for pid in tqdm(builder.val.keys()):
    song_rec, tag_rec = model.recommend(pid, topn_for_song = 10, topn_for_tag=50)
    builder.insert(pid, song_rec, tag_rec)
print(time.time() - s)
print('> use_all_popular :',builder.only_base)

# evaluation
res_file_name = os.path.join(default_dir, r"results/results_w2v.json")
write_json(builder.answers, res_file_name)
evaluator.evaluate(os.path.join(default_dir, r"answers/val_answers.json"), res_file_name)