from drail.features.feature_extractor import FeatureExtractor
from drail.features import utils

import json
import numpy as np
import re

class MusicFE(FeatureExtractor):

    def __init__(self, artist_map, doc2vec_fname,
                 random_state):
        super(MusicFE, self).__init__()
        self.artist_map = artist_map
        self.doc2vec_fname = doc2vec_fname
        np.random.seed(random_state)

    def build(self):
        self.inv_artist = {v: k for k, v in self.artist_map.iteritems()}

        if self.doc2vec_fname:
            self.num_vectors, self.doc2vec_size, self.doc2vec_dic =\
                utils.embeddings_dictionary(self.doc2vec_fname)

        self.artist_idx = {}
        for i, artist in enumerate(self.artist_map):
            self.artist_idx[artist] = i
            # intialize OOV artists
            if artist not in self.doc2vec_dic:
                self.doc2vec_dic[artist] =\
                    np.random.uniform(-0.0025, 0.0025, self.doc2vec_size)

    def artist_1(self, rule_grd):
        doc = rule_grd.get_body_predicates("Artist")[0]['arguments'][0]
        doc = self.inv_artist[doc]
        return self.doc2vec_dic[doc]

    def artist_2(self, rule_grd):
        doc = rule_grd.get_body_predicates("Artist")[1]['arguments'][0]
        doc = self.inv_artist[doc]
        return self.doc2vec_dic[doc]

    def played_1(self, rule_grd):
        doc = rule_grd.get_body_predicates("PlayedNext")[0]['arguments'][1]
        doc = self.inv_artist[doc]
        return self.doc2vec_dic[doc]

    def played_2(self, rule_grd):
        doc = rule_grd.get_body_predicates("PlayedNext")[0]['arguments'][2]
        doc = self.inv_artist[doc]
        return self.doc2vec_dic[doc]

    def extract_multiclass_head(self, instance_grd):
        artist = instance_grd.get_head_predicate()['arguments'][2]
        doc = self.inv_artist[doc]
        return self.artist_idx[artist]

