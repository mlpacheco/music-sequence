import os
import csv
import random

from drail.model.rule import RuleGrounding
from drail.model.label import LabelType

class MusicDB(object):

    def __init__(self):
        self.artists = set([])
        self.played_after = {}
        self.played_pos = {}
        self.users = set([])
        random.seed(42)

    def load_data(self, dataset_path):
        fpath = os.path.join(dataset_path, "artist_map.csv")
        with open(fpath) as f:
            reader = csv.reader(f)
            for row in reader:
                self.artists.add(row[0])
                self.played_after[row[0]] = []

        fpath = os.path.join(dataset_path, "ftrain.csv")
        with open(fpath) as f:
            reader = csv.reader(f)
            for row in reader:
                self.users.add(row[0])
                self.played_pos[row[0]] = []
                for i in range(1, len(row)):
                    self.played_pos[row[0]].append(row[i])
                    if i+1 < len(row):
                        self.played_after[row[i]].append(row[i+1])

        print "Users:", len(self.users)
        print "Artists", len(self.artists)

    def add_filters(self, train, dev, test):
        self.filters = {}
        self.filters['train'] = train
        self.filters['dev'] = dev
        self.filters['test'] = test

    def set_artistmap(self, artist_map):
        self.artist_map = artist_map

    def predict(self, relevant, most_similar):
        ret = []
        for (user, prev, curr) in relevant:
            for artist in most_similar[user]:
                mapped = self.artist_map[artist]
                PlayedUntil = {
                    "name": "PlayedUntil",
                    "arguments": [user, prev],
                    "ttype": None,
                    "obs": True,
                    "isneg": False,
                    "target_pos": None
                }

                Artist = {
                    "name": "Artist",
                    "arguments": [mapped],
                    "ttype": None,
                    "obs": True,
                    "isneg": False,
                    "target_pos": None
                }

                PlayedNext = {
                    "name": "PlayedNext",
                    "arguments": [user, prev, mapped],
                    "ttype": None,
                    "obs": False,
                    "isneg": False,
                    "target_pos": None
                }

                istruth = (artist == self.played_pos[user][curr-1])

                body = [PlayedUntil, Artist]
                head = PlayedNext
                rg = RuleGrounding(body, head, None, False, head['arguments'][1], istruth)
                rg.build_predicate_index()
                rg.build_body_predicate_dic()
                ret.append(rg)

        print "Predict rules", len(ret)
        return ret

    def until_bin(self, istrain, isneg, filters, split_class, instance_id):
        ret = []; relevant = []
        if len(filters) > 0:
            relevant += self.filters[filters[0]]
        else:
            relevant = [(u, 26, 27) for u in self.users]

        if istrain:
            for (user, prev, curr) in relevant:

                PlayedUntil = {
                    "name": "PlayedUntil",
                    "arguments": [user, prev],
                    "ttype": None,
                    "obs": True,
                    "isneg": False,
                    "target_pos": None
                }

                if not isneg:
                    nextone = self.artist_map[self.played_pos[user][curr-1]]

                    Artist = {
                        "name": "Artist",
                        "arguments": [nextone],
                        "ttype": None,
                        "obs": True,
                        "isneg": False,
                        "target_pos": None
                    }

                    PlayedNext = {
                        "name": "PlayedNext",
                        "arguments": [user, prev, nextone],
                        "ttype": None,
                        "obs": False,
                        "isneg": False,
                        "target_pos": None
                    }

                    body = [PlayedUntil, Artist]
                    head = PlayedNext
                    rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                    rg.build_predicate_index()
                    rg.build_body_predicate_dic()
                    ret.append(rg)
                else:
                    lastone = self.played_pos[user][prev-1]
                    negatives = list(self.artists - set(self.played_after[lastone]))
                    random.shuffle(negatives)
                    for nextone in negatives[:1]:
                        nextone = self.artist_map[nextone]
                        Artist = {
                            "name": "Artist",
                            "arguments": [nextone],
                            "ttype": None,
                            "obs": True,
                            "isneg": False,
                            "target_pos": None
                        }

                        PlayedNext = {
                            "name": "PlayedNext",
                            "arguments": [user, prev, nextone],
                            "ttype": None,
                            "obs": False,
                            "isneg": True,
                            "target_pos": None
                        }

                        body = [PlayedUntil, Artist]
                        head = PlayedNext
                        rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                        rg.build_predicate_index()
                        rg.build_body_predicate_dic()
                        ret.append(rg)
        return ret



