from collections import Counter
from itertools import chain
from tqdm import tqdm
from weighted_ratings import Ratings
import time

from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,remove_stopwords, stem_text,strip_multiple_whitespaces

class Playlist2Vec:
    def __init__(self, train_path, val_path):
        """
        Word2Vec Based Song/Tag Recommender
        """
        self.train = load_json(train_path)
        self.val = load_json(val_path)
        self.data = load_json(train_path) + load_json(val_path)

        print('*** Build Vocab ***')
        self.build_vocab()

    def build_vocab(self):
        self.id_to_songs = {}
        self.id_to_tags = {}
        self.id_to_title = {}
        self.corpus = {}

        filters = [remove_stopwords, stem_text, strip_punctuation, strip_multiple_whitespaces]

        for ply in self.data:
            pid = str(ply['id'])
            self.id_to_songs[pid] = [*map(str, ply['songs'])]  # list
            self.id_to_tags[pid] = [*map(str, ply['tags'])]  # list
            self.id_to_title[pid] = preprocess_string(ply['plylst_title'], filters)  # list
            ply['tags'].extend(self.id_to_title[pid])

            self.corpus[pid] = self.id_to_songs[pid] + self.id_to_tags[pid] + self.id_to_title[pid]

        self.songs = set(chain.from_iterable(self.id_to_songs.values()))
        self.tags = set(chain.from_iterable(list(self.id_to_tags.values()) + list(self.id_to_title.values())))

        print("> Corpus :", len(self.corpus))
        print(f'> Songs + Tags = {len(self.songs)} + {len(self.tags)} = {len(self.songs) + len(self.tags)}')
        print("> Playlist Id Type :", type(list(self.id_to_songs.keys())[0]), type(list(self.id_to_tags.keys())[0]))

    def build_scores(self):
        self.rating_builder = Ratings(self.data)
        self.ratings = self.rating_builder.build_coo()
        self.ratings_weighted = 5 * self.rating_builder.bm25_weight(self.ratings).tocsr()

    def get_score(self, pid):
        """
        :param pid: playlist id
        :return: filtered items and scores ( default : bm25 )
        """
        target = self.ratings_weighted[int(pid)]
        scores = target.data
        iids = target.indices
        songs, song_scores, tags, tag_scores = [], [], [], []

        for iid, score in zip(iids, scores):
            if iid >= self.rating_builder.num_song:
                tag = self.rating_builder.id2tag[iid - self.rating_builder.num_song]
                if self.w2v_model.wv.vocab.get(tag):
                    tags.append(tag)
                    tag_scores.append(score)
            else:
                song = str(iid)
                if self.w2v_model.wv.vocab.get(song):
                    songs.append(song)
                    song_scores.append(score)

        return songs, song_scores, tags, tag_scores

    def register_w2v(self, w2v_model):
        self.w2v_model = w2v_model

    def train_w2v(self, min_count=3, size=128, window=250, sg=1, workers=1):
        # workers = 1 ; for consistency
        start = time.time()
        self.w2v_model = Word2Vec(sentences=list(self.corpus.values()), min_count=min_count, size=size, window=window,
                                  sg=sg, workers=workers)
        print(f'> running time : {time.time() - start:.3f}')

    def get_weighted_embedding(self, items, normalize=True, scores=None):
        """
        :param items: list of songs/tags
        :param normalize: if True, embedding will be divided by sum of scores or length of items
        :param scores: weights ( default : bm25 )
        :return: embedding
        """
        items = [str(item) for item in items if self.w2v_model.wv.vocab.get(item)]

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

    def build_p2v(self, normalize_song=True, normalize_tag=True, normalize_title=True,
                  song_weight=1, tag_weight=1,use_bm25=True):
        """
        :param normalize_song: if True, song embedding will be divided sum of scores.
        :param normalize_tag: if True, tag embedding will be divided sum of scores.
        :param normalize_title: if True, title embedding will be divided sum of scores.
        :param song_weight: float
        :param tag_weight: float
        :param use_bm25: if True, scores will be bm25 weights else 1
        """
        start = time.time()
        pids = []
        playlist_embedding = []
        if use_bm25:
            self.build_scores()

        for pid in tqdm(self.corpus.keys()):
            ply_embedding = 0

            if use_bm25:
                songs, song_scores, tags, tag_scores = self.get_score(pid)
                ply_embedding += song_weight * self.get_weighted_embedding(songs, normalize=normalize_song,
                                                                           scores=song_scores)
                ply_embedding += tag_weight * self.get_weighted_embedding(tags, normalize=normalize_tag,
                                                                          scores=tag_scores)
            else:
                ply_embedding += song_weight * self.get_weighted_embedding(self.id_to_songs[pid], normalize_song)
                ply_embedding += tag_weight * self.get_weighted_embedding(self.id_to_tags[pid], normalize_tag)
                ply_embedding += tag_weight * self.get_weighted_embedding(self.id_to_title[pid], normalize_title)

            if type(ply_embedding) != int:  # 한 번이라도 update 되었다면
                pids.append(str(pid))  # ! string !
                playlist_embedding.append(ply_embedding)

        self.p2v_model = WordEmbeddingsKeyedVectors(self.w2v_model.vector_size)
        self.p2v_model.add(pids, playlist_embedding)

        print(f'> running time : {time.time() - start:.3f}')
        print(f'> Register (ply update) : {len(pids)} / {len(self.id_to_songs)}')
        val_ids = set([str(p["id"]) for p in self.val])
        print(f'> Only {len(val_ids - set(pids))} of validation set ( total : {len(val_ids)} ) can not find similar playlist in train set.')

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
    import pickle
    import os

    default_dir = r'C:\Users\haeyu\PycharmProjects\KakaoArena\arena_data'

    train_path = os.path.join(default_dir, r'orig/train.json')
    val_que_path = os.path.join(default_dir, r'questions/val_questions.json')

    model = Playlist2Vec(train_path, val_que_path)

    with open(os.path.join(default_dir, 'model','w2v_128.pkl'), 'rb') as f:
        w2v_model = pickle.load(f)

    model.register_w2v(w2v_model)
    model.build_p2v(normalize_song = True, normalize_tag = True, normalize_title = True , song_weight = 1, tag_weight = 1 ,use_bm25 = False)
    model.build_p2v(normalize_song = False, normalize_tag = True, normalize_title = True , song_weight = 1, tag_weight = 1 ,use_bm25 = True)