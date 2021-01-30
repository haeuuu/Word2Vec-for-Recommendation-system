import copy
import numpy as np
from itertools import chain
from scipy.sparse import coo_matrix, csr_matrix
from math import log

class Ratings:
    """
    train, test(또는 val) set을 받아서 tag2id dict를 구성하고 ALS 학습을 위한 coo matrix를 생성한다.
    """

    def __init__(self, data):
        self.data = copy.deepcopy(data)  # [{'pid', 'songs', 'tags', }, ... ]
        self.num_song = 707989

        self._get_tag2id()

    def _get_tag2id(self):
        """
        tag의 id는 0부터 시작한다. 단 coo matrix를 구성할 때는 0 + self.num_song 부터 시작한다.
        """
        tag_set = set(chain.from_iterable(ply['tags'] for ply in self.data))
        self.num_tag = len(tag_set)
        self.tag2id = {x: i for i, x in enumerate(sorted(tag_set))}
        self.id2tag = {i: x for x, i in self.tag2id.items()}

    def get_raw_tag(self, tids):
        return [self.id2tag[tid] for tid in tids]

    def build_coo(self, w2v_model):
        """
        user id와 item id가 연속적이지 않다면 0인 row가 포함된다.
        ratings의 크기는 (max(uid)+1, max(iid)+1)이 된다.
        """
        pids = []
        iids = []

        for ply in self.data:
            trained_songs = [song for song in ply['songs'] if w2v_model.wv.vocab.get(str(song))]
            trained_tags = [tag for tag in ply['tags'] if w2v_model.wv.vocab.get(tag)]

            rep = len(trained_songs) + len(trained_tags)

            iids.extend(trained_songs)
            iids.extend(self.tag2id[t] + self.num_song for t in
                        trained_tags)  # tag2id는 0부터 시작하므로 self.tag2id[t] + self.num_song 임에 주의

            pids.extend([ply['id']] * rep)

        scores = [1] * len(pids)

        ratings = csr_matrix((np.array(scores, dtype=np.float32),
                              (np.array(pids),
                               np.array(iids))),
                             shape=(max(pids) + 1, self.num_song + self.num_tag))

        return ratings

    def bm25_weight(self, X, K1=100, B=0.9):
        """
        Reference : https://github.com/benfred/implicit
        Weighs each row of a sparse matrix X  by BM25 weighting
        """
        # calculate idf per term (user)
        X = coo_matrix(X)

        N = float(X.shape[0])
        idf = np.log(N) - np.log1p(np.bincount(X.col))

        # calculate length_norm per document (artist)
        row_sums = np.ravel(X.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length

        # weight matrix rows by bm25
        X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]

        return X

class Node:
    def __init__(self, value):
        self.value = value
        self.children = {}
        self.is_terminal = False

class Filter:
    def __init__(self, items):
        self.items = items
        self.head = Node(None)

        print("DB 구성중입니다 ...")
        for item in items:
            self.insert(item)
        print("입력 완료 ...")

    def insert(self, query):
        curr_node = self.head

        for q in query:
            if curr_node.children.get(q) is None:
                curr_node.children[q] = Node(q)
            curr_node = curr_node.children[q]
        curr_node.is_terminal = query

    def extract(self, query, biggest_token = True):
        """
        :param query: str(word)
        :param biggest_token: True인 경우, word에서 찾을 수 있는 가장 큰 tag를 return.
        :return: list of tags
        """
        start, end = 0,0
        query += '*'
        curr_node = self.head
        prev_node = self.head

        extracted_tags = []
        while end < len(query):
            curr_node = curr_node.children.get(query[end])
            if curr_node is None:
                if biggest_token and prev_node.is_terminal:
                    extracted_tags.append(prev_node.is_terminal)
                    start = end-1
                    prev_node = self.head
                start += 1
                end = start
                curr_node = self.head
            else:
                if not biggest_token and curr_node.is_terminal:
                    extracted_tags.append(curr_node.is_terminal)
                elif curr_node.is_terminal:
                    prev_node = curr_node
                end += 1

        return  extracted_tags

    def extract_from_title(self, title, biggest_token = True):
        """
        :param title: str(title)
        :param biggest_token: True인 경우, word에서 찾을 수 있는 가장 큰 tag를 return.
        :return: list of tags
        """
        tags = []
        for word in title.split():
            tags.extend(self.extract(word,biggest_token))
        return tags