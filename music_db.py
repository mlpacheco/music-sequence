import os
import csv

from drail.model.rule import RuleGrounding
from drail.model.label import LabelType

class MusicDB(object):

    def __init__(self):
        self.played_pos = {}
        self.artists = set([])

    def load_data(self, dataset_path):
        fpath = os.path.join(dataset_path, "ftrain.csv")
        with open(fpath) as f:
            reader = csv.reader(f)
            for row in reader:
                self.played_pos[row[0]] = []
                for i in range(1, len(row)):
                    self.played_pos[row[0]].append(row[i])

        fpath = os.path.join(dataset_path, "artist_map.csv")
        with open(fpath) as f:
            reader = csv.reader(f)
            for row in reader:
                self.artists.add(row[0])

        print "Users:", len(self.played_pos)
        print "Artists", len(self.artists)

    def add_filters(self, train, dev, test):
        self.filters = {}
        self.filters['train'] = train
        self.filters['dev'] = dev
        self.filters['test'] = test

    def curr_next(self, istrain, isneg, filters, split_class, instance_id):
        # TO-DO: deal with filters

        if instance_id is None:
            instances = self.played_pos.keys()
        else:
            instances = set([instance_id])

        ret = []
        if istrain:
            for user in instances:
                for i, artist in enumerate(self.played_pos[user]):

                    InPlaylist = {
                        "name": "InPlaylist",
                        "arguments": [user, artist],
                        "ttype": None,
                        "obs": True,
                        "isneg": False,
                        "target_pos": None
                    }

                    if i + 1 >= len(self.played_pos[user]):
                        continue

                    nextone = self.played_pos[user][i+1]

                    PlayedNext = {
                        "name": "PlayedNext",
                        "arguments": [user, artist, nextone],
                        "ttype": LabelType.Multiclass,
                        "obs": False,
                        "isneg": False,
                        "target_pos": 2
                    }

                    body = [InPlaylist]
                    head = PlayedNext
                    rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                    rg.build_predicate_index()
                    rg.build_body_predicate_dic()
                    ret.append(rg)
        return ret

    def prev_curr_next(self, istrain, isneg, filters, split_class, instance_id):
        # TO-DO: deal with filters

        if instance_id is None:
            instances = self.played_pos.keys()
        else:
            instances = set([instance_id])

        ret = []
        if istrain:
            for user in instances:
                for i, prev in enumerate(self.played_pos[user]):

                    if (i + 1) >= len(self.played_pos[user]) or\
                       (i + 2) >= len(self.played_pos[user]):
                        continue

                    curr = self.played_pos[user][i+1]

                    PlayedNext = {
                        "name": "PlayedNext",
                        "arguments": [user, prev, curr],
                        "ttype": LabelType.Multiclass,
                        "obs": False,
                        "isneg": False,
                        "target_pos": 2
                    }

                    nextone = self.played_pos[user][i+2]

                    PlayedNext_ = {
                        "name": "PlayedNext",
                        "arguments": [user, curr, nextone],
                        "ttype": LabelType.Multiclass,
                        "obs": False,
                        "isneg": False,
                        "target_pos": 2
                    }

                    body = [PlayedNext]
                    head = PlayedNext_
                    rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                    rg.build_predicate_index()
                    rg.build_body_predicate_dic()
                    ret.append(rg)


        return ret


