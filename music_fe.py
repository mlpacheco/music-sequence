from drail.features.feature_extractor import FeatureExtractor
from drail.features import utils

import json
import numpy as np
import re

class MusicFE(FeatureExtractor):

    def __init__(self, artist_map_fname):
        super(MusicFE, self).__init__()
        self.artist_map_fname = artist_map_fname

    def build(self):
        with open(self.artist_map_fname) as f:
            artist_map = json.load(f)

        self.artist_idx = {}
        for i, artist in enumerate(artist_map):
            self.artist_idx[artist] = i

    def current_artist(self, rule_grd):
        doc = rule_grd.get_body_predicates("InPlaylist")[0]['arguments'][1]
        return [1]

    def artist_1(self, rule_grd):
        doc = rule_grd.get_body_predicates("PlayedNext")[0]['arguments'][1]
        return [1]

    def artist_2(self, rule_grd):
        doc = rule_grd.get_body_predicates("PlayedNext")[0]['arguments'][2]
        return [1]

    def extract_multiclass_head(self, instance_grd):
        artist = instance_grd.get_head_predicate()['arguments'][2]
        return self.artist_idx[artist]

