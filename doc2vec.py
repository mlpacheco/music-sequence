from gensim.models.doc2vec import Doc2Vec
import json
import gensim
import gensim.utils as ut
from collections import namedtuple
import random

ArtistDocument = namedtuple('ArtistDocument', 'words tags index')

def read_documents(documents, stemmer=0):
    alldocs = []
    allindex={}

    for doc in documents:
        text = documents[doc]

        if stemmer == 1:
            text = gensim.parsing.stem_text(text)
        else:
            text = text.lower()

        words = ut.to_unicode(text).split()
        tags = [doc] # ID of each document for doc2vec model
        index = len(alldocs)

        allindex[doc] = index # mapping from doc ID to index, start from 0
        alldocs.append(ArtistDocument(words, tags, index))

    return alldocs, allindex

def train_doc2vec(doc_list=None, buildvoc=1, passes=20, dm=0,
                 size=100, dm_mean=0, window=5, hs=1, negative=5,
                 min_count=1, workers=4, random_state=42):
    random.seed(random_state)
    model = Doc2Vec(dm=dm, size=size, dm_mean=dm_mean, window=window,
                    hs=hs, negative=negative, min_count=min_count, workers=workers) #PV-DBOW
    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        random.shuffle(doc_list)  # shuffling gets best results

        model.train(doc_list)

    return model

def main():
    embedding = {}

    # some parameters
    numFea = 300
    cores = 4
    random_state = 42
    passes = 100
    dm = 0

    # Load documents into memory
    with open("summaries.json") as f:
        documents = json.load(f)

    print "Artists found:", len(documents)

    alldocs, allindex = read_documents(documents)
    doc_list = alldocs[:] # for future reshuffling

    doc2vec_model = train_doc2vec(doc_list, workers=cores, size=numFea,
                                  dm=dm, passes=passes, min_count=3,
                                  random_state=42)

    for doc in doc_list:
        vector = doc2vec_model.docvecs[doc.tags[0]]
        embedding[doc.tags[0]] = vector

    dummy_key = embedding.keys()[0]
    f_doc2vec = open('doc2vec.bin', 'wb')
    f_doc2vec.write(str(len(embedding)))
    f_doc2vec.write(' ')
    f_doc2vec.write(str(len(embedding[dummy_key])))
    f_doc2vec.write('\n')

    for key in embedding:
        f_doc2vec.write(key)
        f_doc2vec.write(' ')
        f_doc2vec.write(embedding[key].tobytes())
        f_doc2vec.write('\n')

    f_doc2vec.close()
if __name__ == "__main__":
    main()
