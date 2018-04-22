import os
import csv
import random

from drail.model.rule import RuleGrounding
from drail.model.label import LabelType

class MusicDB(object):

    def __init__(self):
        self.played_pos = {}
        self.artists = set([])
        self.played_after = {}
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
                self.played_pos[row[0]] = []
                for i in range(1, len(row)):
                    self.played_pos[row[0]].append(row[i])

                    if i+1 < len(row):
                        self.played_after[row[i]].append(row[i+1])

        print "Users:", len(self.played_pos)
        print "Artists", len(self.artists), len(self.played_after)

    def add_filters(self, train, dev, test):
        self.filters = {}
        self.filters['train'] = train
        self.filters['dev'] = dev
        self.filters['test'] = test

    def set_artistmap(self, artist_map):
        self.artist_map = artist_map

    def predict_next(self, tuples):
        pass

    def curr_next_bin(self, istrain, isneg, filters, split_class, instance_id):
        relevant = []
        if len(filters) > 0:
            relevant += self.filters[filters[0]]
        relevant = set(relevant)

        ret = []
        if istrain:
            for artist in self.artists:

                if len(relevant) > 0 and artist not in relevant:
                    continue

                Artist = {
                    "name": "Artist",
                    "arguments": [self.artist_map[artist]],
                    "ttype": None,
                    "obs": True,
                    "isneg": False,
                    "target_pos": None
                }

                if not isneg:
                    for nextone in self.played_after[artist]:
                        Artist_ = {
                            "name": "Artist",
                            "arguments": [self.artist_map[nextone]],
                            "ttype": None,
                            "obs": True,
                            "isneg": False,
                            "target_pos": None
                        }
                        PlayedNext = {
                            "name": "PlayedNext",
                            "arguments": [self.artist_map[artist],
                                          self.artist_map[nextone]],
                            "ttype": None,
                            "obs": False,
                            "isneg": False,
                            "target_pos": None
                        }

                        body = [Artist, Artist_]
                        head = PlayedNext
                        rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                        rg.build_predicate_index()
                        rg.build_body_predicate_dic()
                        ret.append(rg)
                else:
                    negative_samples = list(set(self.artists) - set(self.played_after[artist]))
                    random.shuffle(negative_samples)
                    for nextone in negative_samples[:10]:
                        Artist_ = {
                            "name": "Artist",
                            "arguments": [self.artist_map[nextone]],
                            "ttype": None,
                            "obs": True,
                            "isneg": False,
                            "target_pos": None
                        }
                        PlayedNext = {
                            "name": "PlayedNext",
                            "arguments": [self.artist_map[artist],
                                          self.artist_map[nextone]],
                            "ttype": None,
                            "obs": False,
                            "isneg": True,
                            "target_pos": None
                        }

                        body = [Artist, Artist_]
                        head = PlayedNext
                        rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                        rg.build_predicate_index()
                        rg.build_body_predicate_dic()
                        ret.append(rg)

        return ret

    def curr_next(self, istrain, isneg, filters, split_class, instance_id):
        # TO-DO: deal with filters
        if instance_id is None:
            instances = self.played_pos.keys()
        else:
            instances = set([instance_id])
        relevant = []
        if len(filters) > 0:
            relevant += self.filters[filters[0]]
        relevant = set(relevant)

        ret = []
        if istrain:
            for user in instances:
                for i, artist in enumerate(self.played_pos[user]):

                    InPlaylist = {
                        "name": "InPlaylist",
                        "arguments": [user, self.artist_map[artist]],
                        "ttype": None,
                        "obs": True,
                        "isneg": False,
                        "target_pos": None
                    }

                    if i + 1 >= len(self.played_pos[user]):
                        continue

                    nextone = self.played_pos[user][i+1]

                    if len(relevant) > 0 and \
                       (user, artist, nextone) not in relevant:
                        continue

                    PlayedNext = {
                        "name": "PlayedNext",
                        "arguments": [user,
                                      self.artist_map[artist],
                                      self.artist_map[nextone]],
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

    def prev_curr_next_bin(self, istrain, isneg, filters, split_class, instance_id):
        if instance_id is None:
            instances = self.played_pos.keys()
        else:
            instances = set([instance_id])
        relevant = []
        if len(filters) > 0:
            relevant += self.filters[filters[0]]
        relevant = set(relevant)

        ret = []
        if istrain:
            for user in instances:
                for i, prev in enumerate(self.played_pos[user]):

                    if (i + 1) >= len(self.played_pos[user]) or\
                       (i + 2) >= len(self.played_pos[user]) or\
                       (len(relevant) > 0 and 
                        (user, self.played_pos[user][i+1],
                        self.played_pos[user][i+2]) not in relevant):
                        continue

                    curr = self.played_pos[user][i+1]

                    PlayedNext = {
                        "name": "PlayedNext",
                        "arguments": [user, self.artist_map[prev],
                                            self.artist_map[curr]],
                        "ttype": None,
                        "obs": True,
                        "isneg": False,
                        "target_pos": None
                    }


                    if not isneg:
                        nextone = self.played_pos[user][i+2]

                        Artist = {
                            "name": "Artist",
                            "arguments": [self.artist_map[nextone]],
                            "ttype": None,
                            "obs": True,
                            "isneg": False,
                            "target_pos": None
                        }

                        PlayedNext_ = {
                            "name": "PlayedNext",
                            "arguments": [self.artist_map[curr],
                                          self.artist_map[nextone]],
                            "ttype": None,
                            "obs": False,
                            "isneg": False,
                            "target_pos": None
                        }

                        body = [PlayedNext, Artist]
                        head = PlayedNext_
                        rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                        rg.build_predicate_index()
                        rg.build_body_predicate_dic()
                        ret.append(rg)
                        #print "POS", rg

                    else:
                        negative_samples = list(set(self.artists) - set(self.played_after[curr]))
                        random.shuffle(negative_samples)
                        for nextone in negative_samples[:5]:
                            Artist = {
                                "name": "Artist",
                                "arguments": [self.artist_map[nextone]],
                                "ttype": None,
                                "obs": True,
                                "isneg": False,
                                "target_pos": None
                            }

                            PlayedNext_ = {
                                "name": "PlayedNext",
                                "arguments": [self.artist_map[curr],
                                              self.artist_map[nextone]],
                                "ttype": None,
                                "obs": False,
                                "isneg": True,
                                "target_pos": None
                            }

                            body = [PlayedNext, Artist]
                            head = PlayedNext_
                            rg = RuleGrounding(body, head, None, False, head['arguments'][1], True)
                            rg.build_predicate_index()
                            rg.build_body_predicate_dic()
                            ret.append(rg)
                            #print "NEG", rg
                        #exit()


        return ret


