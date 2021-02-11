import os
import re
import time
import pickle
import numpy as np

from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm

from .weighted_ratings import Ratings
from .extract_tags import TagExtractor
from .util import load_json

from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

class Playlist2Vec:
    def __init__(self, train_path, val_path, limit = 2):
        """
        Word2Vec Based Song/Tag Recommender
        """
        self.train = load_json(train_path)
        self.val = load_json(val_path)
        self.data = load_json(train_path) + load_json(val_path)

        print('Build Vocab ...')
        self.build_vocab(limit)

    def build_vocab(self, limit):
        self.id_to_songs = {}
        self.id_to_tags = {}
        self.id_to_title = {}
        self.corpus = {}

        self.filter = TagExtractor(limit)
        self.filter.build_by_vocab(set(chain.from_iterable([ply['tags'] for ply in self.data])))

        for ply in tqdm(self.data):
            pid = str(ply['id'])
            self.id_to_songs[pid] = [*map(str, ply['songs'])]  # list
            self.id_to_tags[pid] = [*map(str, ply['tags'])]  # list

            raw_title = re.findall('[0-9a-zA-Z가-힣]+' ,ply['plylst_title'])
            extracted_tags = self.filter.convert(" ".join(raw_title))
            self.id_to_title[pid] = extracted_tags
            ply['tags'] = set(self.id_to_title[pid] + ply['tags'])

            self.corpus[pid] = self.id_to_songs[pid] + self.id_to_tags[pid] + self.id_to_title[pid]

        self.songs = set(chain.from_iterable(self.id_to_songs.values()))
        self.tags = set(chain.from_iterable(list(self.id_to_tags.values()) + list(self.id_to_title.values())))

        print("> Corpus :", len(self.corpus))
        print(f'> Songs + Tags = {len(self.songs)} + {len(self.tags)} = {len(self.songs) + len(self.tags)}')
        print("> Playlist Id Type :", type(list(self.id_to_songs.keys())[0]), type(list(self.id_to_tags.keys())[0]))

    def build_bm25(self):

        print('Build bm25 ...')
        self.rating_builder = Ratings(self.data)
        self.ratings = self.rating_builder.build_coo(self.w2v_model)
        self.ratings_weighted = 5 * self.rating_builder.bm25_weight(self.ratings).tocsr()
        self.idf = self.rating_builder.idf_weight(self.ratings)

        self.bm25 = defaultdict(lambda: {'songs': [[], []], 'tags': [[],
                                                                     []]})  # {pid : {'songs' : [ [song1, song2, ... ], [score1, score2, ... ] ], 'tags' : [ [tag1, tag2, ...], [score1 , score2 , ...] ] } , ... }

        for pid in tqdm(self.corpus.keys()):

            target = self.ratings_weighted[int(pid)]
            scores = target.data
            iids = target.indices

            for iid, score in zip(iids, scores):
                if iid >= self.rating_builder.num_song:
                    tag = self.rating_builder.id2tag[iid - self.rating_builder.num_song]
                    self.bm25[pid]['tags'][0].append(tag)
                    self.bm25[pid]['tags'][1].append(score)
                else:
                    song = str(iid)
                    self.bm25[pid]['songs'][0].append(song)
                    self.bm25[pid]['songs'][1].append(score)

    def build_consistency(self, topn=3):
        """
        co-occurrence를 계산하고 consistency를 얻는다!
        """

        print('Calculate co-occurrence ...')
        co_occur = defaultdict(Counter)
        for items in tqdm(self.corpus.values()):
            cnt = Counter(items)
            for curr in items:
                co_occur[curr].update(cnt)

        for tag, cnt in tqdm(co_occur.items()):
            del cnt[tag]

        print('Get consistency ...')
        self.consistency = {}
        for query in tqdm(co_occur.keys()):
            sims = []
            for tag, sim in co_occur[query].most_common(topn):
                try:
                    sims.append((tag, self.w2v_model.similarity(query, tag)))
                except KeyError:
                    continue

            sims_mean = sum([s for t, s in sims[:topn]]) / topn
            exp_mean = np.exp(sims_mean * 5)
            self.consistency[query] = exp_mean

        del co_occur

    def register_w2v(self, w2v_model_path):
        with open(w2v_model_path, 'rb') as f:
            self.w2v_model = pickle.load(f)

    def train_w2v(self, min_count=3, size=128, window=250, sg=1, workers=1):
        # workers = 1 ; for consistency
        start = time.time()
        self.w2v_model = Word2Vec(sentences=list(self.corpus.values()), min_count=min_count, size=size, window=window,
                                  sg=sg, workers=workers)
        print(f'> running time : {time.time() - start:.3f}')

    def get_weighted_embedding(self, items, normalize=True, scores=None):
        """
        items의 embedding을 scores에 따라 weighted sum한 결과를 return합니다.
        :param items: list of songs/tags
        :param normalize: if True, embedding vector will be divided by sum of scores or length of items
        :param mode: bm25 or consistency ( default : bm25 )
        :return: embedding vector
        """
        if not items:
            return 0

        if scores is None:
            scores = [1] * len(items)

        embedding = 0
        for item, score in zip(items, scores):
            embedding += score * self.w2v_model.wv.get_vector(item)

        if normalize:
            embedding /= sum(scores)

        return embedding

    def build_p2v(self, normalize_song=True, normalize_tag=True,
                  song_weight=1, tag_weight=1, mode='consistency'):
        """
        :param normalize_song: if True, song embedding will be divided sum of scores.
        :param normalize_tag: if True, tag embedding will be divided sum of scores.
        :param normalize_title: if True, title embedding will be divided sum of scores.
        :param song_weight: float
        :param tag_weight: float
        """
        start = time.time()
        pids = []
        playlist_embedding = []

        if mode == 'bm25':
            self.build_bm25()
        elif mode == 'consistency':
            self.build_consistency()
        else:
            songs_score = None
            tags_score = None

        for pid in tqdm(self.corpus.keys()):
            ply_embedding = 0

            if mode == 'bm25':
                songs, songs_score = self.bm25[pid]['songs']
                tags, tags_score = self.bm25[pid]['tags']
            else:
                songs = [song for song in self.id_to_songs[pid] if self.w2v_model.wv.vocab.get(song)]
                tags = [tag for tag in self.id_to_tags[pid] + self.id_to_title[pid] if self.w2v_model.wv.vocab.get(tag)]

                if mode == 'consistency':
                    songs_score = [self.consistency[song] for song in songs]
                    tags_score = [self.consistency[tag] for tag in tags]

            ply_embedding += song_weight * self.get_weighted_embedding(songs, normalize_song, scores=songs_score)
            ply_embedding += tag_weight * self.get_weighted_embedding(tags, normalize_tag, scores=tags_score)

            if type(ply_embedding) != int:  # 한 번이라도 update 되었다면
                pids.append(str(pid))  # ! string !
                playlist_embedding.append(ply_embedding)

        self.p2v_model = WordEmbeddingsKeyedVectors(self.w2v_model.vector_size)
        self.p2v_model.add(pids, playlist_embedding)

        print(f'> running time : {time.time() - start:.3f}')
        print(f'> Register (ply update) : {len(pids)} / {len(self.id_to_songs)}')
        val_ids = set([str(p["id"]) for p in self.val])
        print(
            f'> Only {len(val_ids - set(pids))} of validation set ( total : {len(val_ids)} ) can not find similar playlist in train set.')

    def recommend(self, pid, topn_for_tag, topn_for_song):
        """
        :param pid: playlist id
        :param topn_for_tag: number of simliar playlists for tag recommendation
        :param topn_for_song: number of simliar playlists for song recommendation
        :return: recommended songs, recommended tags
        """

        if self.p2v_model.vocab.get(str(pid)) is None:
            return [], []
        else:
            ply_embedding = 0

            ply_candidates = self.p2v_model.most_similar(str(pid), topn=max(topn_for_tag, topn_for_song))
            song_candidates = []
            tag_candidates = []

            for cid, _ in ply_candidates[:topn_for_song]:
                song_candidates.extend(self.id_to_songs[str(cid)])
            for cid, _ in ply_candidates[:topn_for_tag]:
                tag_candidates.extend(self.id_to_tags[str(cid)])

            song_most_common = [song for song, _ in Counter(song_candidates).most_common()]
            tag_most_common = [tag for tag, _ in Counter(tag_candidates).most_common()]

            return song_most_common, tag_most_common

if __name__ == '__main__':
    dir = r'C:\Users\haeyu\PycharmProjects\KakaoArena\arena_data'

    train_path = os.path.join(dir, 'orig','train.json')
    val_que_path = os.path.join(dir, 'questions','val_question.json')
    w2v_model_path = os.path.join(dir, 'model','w2v_128.pkl')

    model = Playlist2Vec(train_path, val_que_path)
    model.register_w2v(w2v_model_path)

    # example
    model.build_p2v(normalize_song = True, normalize_tag = True, song_weight = 1, tag_weight = 1, mode = None)
    rec_songs, rec_tags = model.recommend(147332, topn_for_song = 20, topn_for_tag = 50)
    print(rec_tags)
    print(rec_songs[:10])