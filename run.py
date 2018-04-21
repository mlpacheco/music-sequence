from drail.learn.local_learner import LocalLearner
import csv

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

def main():
    rule_file = "rule.dr"
    conf_file = "config.json"

    learner=LocalLearner()
    learner.compile_rules(rule_file)
    db=learner.create_dataset("data/", dbmodule_path=".")

    train, dev, test = get_folds()
    db.add_filters(train, dev, test)

    learner.build_feature_extractors(db,
                                     artist_map_fname="./artist_map.json",
                                     femodule_path=".")

if __name__ == "__main__":
    main()
