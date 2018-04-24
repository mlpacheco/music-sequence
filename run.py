from drail.learn.local_learner import LocalLearner
import csv
import json
from drail.features import utils
from scipy.spatial.distance import cosine
import numpy as np
import ml_metrics as metrics

def get_folds():
    train = []; dev = []; test = []
    fpath = "data/ftrain.csv"
    with open(fpath) as f:
        reader = csv.reader(f)
        for row in reader:
            dev.append((row[0], 27, 28))
            test.append((row[0], 28, 29))

            for i in range(20, 27):
                train.append((row[0], i, i+1))
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

def baseline_prev_artist_sim(artist_map, mode='dev'):
    num_vectors, doc2vec_size, doc2vec_dic =\
        utils.embeddings_dictionary("doc2vec.bin")

    others = doc2vec_dic.values()
    others_id = doc2vec_dic.keys()

    fpath = "data/ftrain.csv"
    with open(fpath) as f:
        reader = csv.reader(f)
        all_ = 0; match = 0; skipped = 0
        for row in reader:
            if row[28] in doc2vec_dic:
                vector = doc2vec_dic[row[28]]
            else:
                skipped += 1
                continue
            distances = [cosine(v, vector) for v in others]
            idx = np.argpartition(distances, 10)
            most_similar = [others_id[i] for i in idx][:10]

            match += int(row[29] in most_similar)
            all_ += 1

    print match, "/", all_
    print "skipped", skipped

def parse_data(artist_map):
    played_pos = {}
    fpath = "data/ftrain.csv"
    with open(fpath) as f:
        reader = csv.reader(f)
        for row in reader:
            played_pos[row[0]] = []
            for i in range(1, len(row)):
                played_pos[row[0]].append(row[i])
    return played_pos

def allprev_artist_sim(artist_map, played_pos, fold, k=10):
    num_vectors, doc2vec_size, doc2vec_dic =\
        utils.embeddings_dictionary("doc2vec.bin")

    others = doc2vec_dic.values()
    others_id = doc2vec_dic.keys()

    most_similar = {}
    for i, (u, prev, target) in enumerate(fold):
        print i
        prevs = [played_pos[u][j] for j in range(prev)]
        vectors = [doc2vec_dic[a] for a in prevs \
                           if a in doc2vec_dic]
        mean = np.mean(vectors, axis=0)
        distances = [cosine(v, mean) for v in others]
        idx = np.argsort(distances)
        most_similar[u] = [others_id[j] for j in idx][:k]

    return most_similar

def measure_baseline_results(played_pos, fold, most_similar):
    mets = []
    for (u, prev, target) in fold:
        actual = [played_pos[u][target-1]]
        predicted = most_similar[u]
        mets.append(metrics.apk(actual, predicted, 10))
    print "Map@10", np.mean(mets)
    return np.mean(mets)

def baseline_allprev_artist_sim(artist_map, k=10):
    played_pos = parse_data(artist_map)
    fold = [(u, 29, 30) for u in played_pos]

    most_similar = allprev_artist_sim(artist_map, played_pos, fold, k)
    out = open("baseline_avg_corrected.csv", 'wb')
    spamwriter = csv.writer(out)
    spamwriter.writerow(['id', 'artist'])

    for u in most_similar:
        wrow = [u] + [" ".join(most_similar[u])]
        spamwriter.writerow(wrow)
    out.close()

def main():
    rule_file = "rule.dr"
    conf_file = "config.json"
    random_state = 42
    artist_map = load_map()

    baseline_allprev_artist_sim(artist_map)
    exit()

    train_u, dev_u, test_u = get_folds()
    played_pos = parse_data(artist_map)
    most_similar_test = allprev_artist_sim(artist_map, played_pos, test_u)
    measure_baseline_results(played_pos, test_u, most_similar_test)


    learner=LocalLearner()
    learner.compile_rules(rule_file)
    db=learner.create_dataset("data/", dbmodule_path=".")

    db.add_filters(train_u, dev_u, test_u)

    db.set_artistmap(artist_map)

    learner.build_feature_extractors(db,
                                     artist_map = artist_map,
                                     doc2vec_fname="./doc2vec.bin",
                                     ftrain="data/ftrain.csv",
                                     random_state=random_state,
                                     femodule_path=".")

    learner.build_models(db, conf_file)

    _ = learner.train(
            db,
            train_filters=["train"],
            dev_filters=["dev"],
            test_filters=["test"]
        )

    test_set = db.predict(test_u, most_similar_dev)
    weights = learner.predict_local_topK(test_set, 0, 0, 0, K=10)

    results = {}
    for rule in weights:
        user = rule.head['arguments'][0]
        pred = rule.head['arguments'][2]

        if user not in results:
            results[user] = {'artists':[], 'weights':[]}

        results[user]['artists'].append(pred)
        results[user]['weights'].append(weights[rule])



    fpath = "data/ftrain.csv"
    with open(fpath) as f:
        reader = csv.reader(f)
        all_ = 0; match = 0

        for row in reader:
            u = row[0]

            idx = np.argpartition(results[u]['weights'], -10)[-10:]
            most_relevant = [results[u]['artists'][i] for i in idx]

            if artist_map[row[29]] in most_relevant:
                match += 1
            all_ += 1
            #print match
        print match, "/", all_

if __name__ == "__main__":
    main()
