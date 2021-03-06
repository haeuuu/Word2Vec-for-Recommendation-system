from util import *
import os

class AnswerBuilder:
    def __init__(self):
        self.goal = {'songs':100,'tags':10}
        self.answers = []
        self.only_base = 0

    def register_questions(self,val_path):
        val = load_json(val_path)
        self.val = {}
        for ply in val:
            self.val[ply['id']] = {'songs':ply['songs'], 'tags':ply['tags'], 'title':ply['plylst_title']}

    def register_answers(self,ans_path):
        ans = load_json(ans_path)
        self.ans = {}
        for ply in ans:
            self.ans[ply['id']] = {'songs':ply['songs'], 'tags':ply['tags']}

    def register_base_results(self, base_results_path):
        base_results = load_json(base_results_path)
        self.base_results = {}
        for ply in base_results:
            self.base_results[ply['id']] = {'songs':ply['songs'], 'tags':ply['tags']}

    def remove_seen(self, id, rec_results, attr):
        filtered_result = []

        for item in rec_results:
            if item not in self.val[id][attr]:
                filtered_result.append(item)
                if len(filtered_result) == self.goal[attr]:
                    break

        assert len(set(filtered_result)) == len(filtered_result)

        return filtered_result

    def fill_up(self, id, rec_results, attr):
        fill = []
        fill_num = self.goal[attr] - len(rec_results)

        for item in self.base_results[id][attr]:
            if item in rec_results or item in self.val[id]:
                continue
            fill.append(item)
            if len(fill) == fill_num:
                break

        rec_results.extend(fill)

        if len(set(rec_results)) != self.goal[attr]:
            print(len(set(rec_results)),attr)
            assert False

    def initialize(self):
        self.answers.clear()
        self.only_base = 0

    def save_answers(self, filename):
        if len(self.answers) == 0:
            print('Please run < self.insert > first.')
            return
        write_json(self.answers,os,path.join(filename))

    def insert(self,id,rec_songs, rec_tags):
        # [preporcessing] sid가 str형인 경우 변경
        rec_songs = list(map(int,rec_songs))

        # [preprocessing] 중복이 있는 경우 제거 ('7','07'이 담기는 경우 추천 결과에 중복이 생김)
        rec_songs = sorted(set(rec_songs), key = rec_songs.index)
        # rec_tags = sorted(set(rec_tags), key = rec_tags.index)

        songs = self.remove_seen(id,rec_songs,'songs')
        tags = self.remove_seen(id,rec_tags,'tags')

        if len(set(songs)) < self.goal['songs']:
            if len(songs) == 0:
                self.only_base += 1
            self.fill_up(id , songs , 'songs')

        if len(set(tags)) < self.goal['tags']:
            self.fill_up(id , tags ,'tags')

        if not len(set(songs)) == self.goal['songs']:
            print(id,'songs : ',len(set(songs)), len(songs), self.goal['songs'])
        if not len(set(tags)) == self.goal['tags']:
            print(id, 'tags : ',len(set(tags)),len(tags), self.goal['tags'])

        self.answers.append({
            'id' : id,
            'songs' : songs,
            'tags' : tags
        })

if __name__ == '__main__':
    # USAGE
    from util import *
    dir = r'C:\Users\haeyu\PycharmProjects\KakaoArena\arena_data'

    train_path = os.path.join(default_dir, r'orig/train.json')
    val_que_path = os.path.join(default_dir, r'questions/val_questions.json')
    val_ans_path = os.path.join(default_dir, r'answers/val_answers.json')
    results_gep_path = os.path.join(default_dir, 'results', 'results_gep.json')

    builder = AnswerBuilder()
    builder.register_questions(val_que_path)
    builder.register_answers(val_ans_path)
    builder.register_base_results(results_gep_path)

    # EXAMPLE
    from tqdm import tqdm

    builder.initialize()
    for pid, dic in tqdm(builder.val.items()):
        # build rec_songs, rec_tags
        # and run 'builder.insert(pid, rec_songs, rec_tags)'
        pass