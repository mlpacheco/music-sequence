from drail.learn.local_learner import LocalLearner
import csv
import json
from drail.features import utils
from scipy.spatial.distance import cosine
import numpy as np

def get_folds():
    train = []; dev = []; test = []
    fpath = "data/ftrain.csv"
    with open(fpath) as f:
        reader = csv.reader(f)
        for row in reader:

            dev.append((row[0], row[26], row[27]))
            test.append((row[0], row[27], row[28]))

            for i in range(0, 26):
                train.append((row[0], row[i], row[i+1]))
    return train, dev, test

def load_map():
    seen_names = {}
    with open('artist_map.json') as f:
        artist_map = json.load(f)

    for artist in artist_map:
        if artist_map[artist] not in seen_names:
            seen_names[artist_map[artist]] = 0
        else:
            seen_names[artist_map[artist]] += 1
            artist_map[artist] =\
                    artist_map[artist] + \
                    "_{0}".format(seen_names[artist_map[artist]]+1)

    return artist_map

def top_similarity(artist_map):
    most_similar = {}
    num_vectors, doc2vec_size, doc2vec_dic =\
        utils.embeddings_dictionary("doc2vec.bin")

    others = doc2vec_dic.values()
    others_id = doc2vec_dic.keys()

    for j, artist in enumerate(artist_map):
        if artist in doc2vec_dic:
            print j, artist_map[artist]
            vector = doc2vec_dic[artist]
            distances = [cosine(v, vector) for v in others]
            idx = np.argpartition(distances, 10)[:11]
            #print artist_map[artist], [artist_map[others_id[i]] for i in idx]
            most_similar[artist] = [others_id[i] for i in idx]
    return most_similar

def baseline_prev_artist_sim(most_similar, artist_map):
    fpath = "data/ftrain.csv"

    with open(fpath) as f:
        reader = csv.reader(f)
        all_ = 0
        for row in reader:
            if row[28] in most_similar:
                print artist_map[row[28]], "|", artist_map[row[29]]
                print [artist_map[x] for x in most_similar[row[28]]]
                match += int(row[29] in most_similar[row[28]])
                all_ += 1
    print match, "/", all_

def baseline_allprev_artist_sim(artist_map):
    out = open("baseline_avg.csv", 'wb')
    spamwriter = csv.writer(out)
    spamwriter.writerow(['id', 'artist'])
    num_vectors, doc2vec_size, doc2vec_dic =\
        utils.embeddings_dictionary("doc2vec.bin")

    others = doc2vec_dic.values()
    others_id = doc2vec_dic.keys()

    fpath = "data/ftrain.csv"
    with open(fpath) as f:
        reader = csv.reader(f)
        all_ = 0; match = 0
        for row in reader:
            prevs = [row[i] for i in range(1, 30)]
            #print "PREVS", [artist_map[a] for a in prevs]
            vectors = [doc2vec_dic[a] for a in prevs \
                               if a in doc2vec_dic]
            mean = np.mean(vectors, axis=0)
            distances = [cosine(v, mean) for v in others]
            idx = np.argpartition(distances, 10)
            #print idx
            most_similar = [others_id[i] for i in idx][:10]

            wrow = [row[0]] + [" ".join(most_similar)]
            spamwriter.writerow(wrow)

            #print artist_map[row[29]]
            #print [artist_map[x] for x in most_similar]

            #match += int(row[29] in most_similar)
            #all_ += 1
    #print match, "/", all_
    out.close()

def main():
    rule_file = "rule.dr"
    conf_file = "config.json"
    random_state = 42
    artist_map = load_map()

    #most_similar = top_similarity(artist_map)
    #sanity_check(most_similar, artist_map)

    baseline_allprev_artist_sim(artist_map)
    exit()

    train_a = artist_map.keys()[0:2800]
    dev_a = artist_map.keys()[2800:2800+600]
    test_a = artist_map.keys()[2800+600:]

    learner=LocalLearner()
    learner.compile_rules(rule_file)
    db=learner.create_dataset("data/", dbmodule_path=".")

    train_u, dev_u, test_u = get_folds()
    #db.add_filters(train_u, dev_u, test_u)
    db.add_filters(train_a, dev_a, test_a)
    db.set_artistmap(artist_map)

    learner.build_feature_extractors(db,
                                     artist_map = artist_map,
                                     doc2vec_fname="./doc2vec.bin",
                                     random_state=random_state,
                                     femodule_path=".")

    learner.build_models(db, conf_file)

    _ = learner.train(
            db,
            train_filters=["train"],
            dev_filters=["dev"],
            test_filters=["test"]
        )


if __name__ == "__main__":
    main()
