import pickle

class Node:
    def __init__(self, value):
        self.value = value
        self.children = {}
        self.is_terminal = False

class TagExtractor:
    """
    vocab에 들어있는 단어만을 추출한다.
    """
    def __init__(self, limit = 2):
        self.head = Node(None)
        self.limit = limit

    def build_by_vocab(self, words):
        for word in words:
            self.insert(word, self.limit)

    def build_by_w2v(self, w2v_model_path):
        with open(w2v_model_path, 'rb') as f:
            w2v_model = pickle.load(f)

        for word in w2v_model.wv.vocab.keys():
            self.insert(word, self.limit)

    def not_satisfied(self, query, limit = 1):
        """
        숫자 또는 알파벳 한글자(의미가 부족한 태그라고 판단)는 False를 return합니다.
        limit이 1 이하인 경우에는 한글의 경우 True를 return합니다. (봄, 팝 등의 태그를 위해)
        limit이 2 이상인 경우에는 길이만을 고려합니다.
        """
        if query.isdigit():
            return True
        if limit <= 1:
            return len(query) <= limit and query.encode().isalpha()
        return len(query) < limit

    def insert(self, query, limit):
        if self.not_satisfied(query, limit):
            return

        curr_node = self.head

        for q in query.lower():
            if curr_node.children.get(q) is None:
                curr_node.children[q] = Node(q)
            curr_node = curr_node.children[q]
        curr_node.is_terminal = query

    def search(self, query, return_value = False):
        curr_node = self.head

        for q in query.lower():
            curr_node = curr_node.children.get(q)
            if curr_node is None:
                return False

        if curr_node.is_terminal:
            if return_value:
                return curr_node.is_terminal
            return True
        return False

    def extract(self, query, biggest_token=True):
        """
        vocab : 잔잔한, 감성
        input : 잔잔한감성 입니당
        return : [ 잔잔한 , 감성 ]
        """
        start, end = 0, 0
        query = query.lower() + '*'
        curr_node = self.head
        prev_node = self.head

        extracted_tags = []
        while end < len(query):
            curr_node = curr_node.children.get(query[end])
            if curr_node is None:
                if biggest_token and prev_node.is_terminal:
                    extracted_tags.append(prev_node.is_terminal)
                    start = end - 1
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

        return extracted_tags

    def convert(self, query, merged = False):
        """
        vocab : Christmas, 잔잔한, 감성
        input : christmas 잔잔한감성 입니당
        extracted : [ Christmas, 잔잔한 , 감성 ] ( if merged == True, '잔잔한감성' 자체가 vocab에 없으므로 추가. )
        return : [ Christmas, 잔잔한 , 감성 , 입니당 ] ( if merged == True, [ Christmas, 잔잔한, 감성, 잔잔한감성, 입니당 ] )
        """
        res = []
        for q in query.split():
            extracted = self.extract(q)
            if extracted:
                res.extend(extracted)
                if merged and not self.search(q):
                        res.append(q.lower())
            else:
                if not self.not_satisfied(q, self.limit):
                    res.append(q)

        return res