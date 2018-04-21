from gensim.models.doc2vec import Doc2Vec
import json

def read_documents(docjson):
    with open(docjson) as f:
        documents = json.load(f)

    for doc in documents:
        text = documents[doc]
        print doc
        print text
        exit()


def train_doc2vec(doc_list=None, buildvoc=1, passes=20, dm=0,
                 size=100, dm_mean=0, window=5, hs=1, negative=5,
                 min_count=1, workers=4):
    model = Doc2Vec(dm=dm, size=size, dm_mean=dm_mean, window=window,
                    hs=hs, negative=negative, min_count=min_count, workers=workers) #PV-DBOW
    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        shuffle(doc_list)  # shuffling gets best results

        model.train(doc_list)

    return model

def main():
    numFea = 300
    cores = 4
    random_state = 42

    read_documents("summaries.json")

    #doc2vec_model = net.trainDoc2Vec(doc_list, workers=cores, size=numFea, dm=dm, passes=passes, min_count=3)

if __name__ == "__main__":
    main()
