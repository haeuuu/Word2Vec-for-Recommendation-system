from collections import Counter
from itertools import chain
from tqdm import tqdm
import time

from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,remove_stopwords, stem_text,strip_multiple_whitespaces

class Playlist2Vec:
    def __init__(self,train, val):
        self.train = train
        self.val = val
        self.data = self.train + self.val

        print('*** Build Vocab ***')
        self.build_vocab()

    def build_vocab(self):
        self.id_to_songs = {}
        self.id_to_tags = {}
        self.id_to_title = {}
        self.id_to_date = {}
        self.corpus = {}

        filters = [remove_stopwords, stem_text,strip_punctuation, strip_multiple_whitespaces]

        for ply in self.data:
            pid = str(ply['id'])
            self.id_to_songs[pid] = [*map(str,ply['songs'])] # list
            self.id_to_tags[pid] = [*map(str,ply['tags'])] # list
            self.id_to_title[pid] = preprocess_string(ply['plylst_title'], filters) # list

            self.corpus[pid] = self.id_to_songs[pid] + self.id_to_tags[pid] + self.id_to_title[pid]

        self.songs = set(chain.from_iterable(self.id_to_songs.values()))
        self.tags = set(chain.from_iterable(self.id_to_tags.values()))

        print("> Corpus :", len(self.corpus))
        print(f'> Songs + Tags = {len(self.songs)} + {len(self.tags)} = {len(self.songs) + len(self.tags)}')
        print("> Playlist Id Type :", type(list(self.id_to_songs.keys())[0]),type(list(self.id_to_tags.keys())[0]))

    def register_w2v(self, w2v_model):
        self.w2v_model = w2v_model

    def train_w2v(self, min_count = 3, size = 128, window = 250, sg = 1,workers = 1):
        # workers = 1 ; for consistency
        start = time.time()
        self.w2v_model = Word2Vec(sentences = list(self.corpus.values()), min_count= min_count , size = size , window = window, sg = sg, workers = workers)
        print(f'> running time : {time.time()-start:.3f}')

    def get_embedding(self,items,normalize = True):
        items = list(filter(lambda x: x in self.w2v_model.wv.vocab, items))

        if len(items) == 0:
            return 0

        embedding = 0
        for item in items:
            embedding += self.w2v_model.wv.get_vector(item)

        if normalize:
            embedding /= len(items)

        return embedding

    def build_p2v(self,normalize_song = True,normalize_tag = True,normalize_title = True):
        start = time.time()
        pids = []
        playlist_embedding = []

        for pid in tqdm(self.corpus.keys()):
            ply_embedding = self.get_embedding(self.id_to_songs[pid],normalize_song)
            ply_embedding += self.get_embedding(self.id_to_tags[pid],normalize_tag)
            ply_embedding += self.get_embedding(self.id_to_title[pid],normalize_title)

            if type(ply_embedding) != int: # 한 번이라도 update 되었다면
                pids.append(str(pid)) # ! string !
                playlist_embedding.append(ply_embedding)

        self.p2v_model = WordEmbeddingsKeyedVectors(self.w2v_model.vector_size)
        self.p2v_model.add(pids,playlist_embedding)

        print(f'> running time : {time.time()-start:.3f}')
        print(f'> Register (ply update) : {len(pids)} / {len(self.id_to_songs)}')
        val_ids = set([str(p["id"]) for p in self.val])
        print(f'> Only {len( val_ids - set(pids) )} of validation set ( total : {len(val_ids)} ) can not find similar playlist in train set.')

    def recommend(self,pid,topn_for_tag = 50, topn_for_song = 10):
        if self.p2v_model.vocab.get(str(pid)) is None:
            return [],[]
        else:
            ply_candidates = self.p2v_model.most_similar(str(pid), topn = max(topn_for_tag, topn_for_song))
            song_candidates = []
            tag_candidates = []

            for cid , _ in ply_candidates[:topn_for_song]:
                song_candidates.extend(self.id_to_songs[str(cid)])
            for cid , _ in ply_candidates[:topn_for_tag]:
                tag_candidates.extend(self.id_to_tags[str(cid)])

            song_most_common = [song for song,_ in Counter(song_candidates).most_common()]
            tag_most_common = [tag for tag,_ in Counter(tag_candidates).most_common()]

            return song_most_common, tag_most_common

if __name__ == '__main__':
    from util import get_data
    import pickle
    import os

    dir = r'C:\Users\haeyu\PycharmProjects\KakaoArena\arena_data'

    train, val_que, _ = get_data(dir, test=True)
    model = Playlist2Vec(train, val_que)

    with open(os.path.join(dir, 'model','w2v_128.pkl'), 'rb') as f:
        w2v_model = pickle.load(f)

    model.register_w2v(w2v_model)
    model.build_p2v()