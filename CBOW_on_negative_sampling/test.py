"""
测试word_embedding效果
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

f = open('word_embedding.txt')
f.readline()
all_embeddings = []
all_words = []
word2id = dict()
for i, line in enumerate(f):
    line = line.strip().split(' ')
    word = line[0]
    embedding = [float(x) for x in line[1:]]
    assert len(embedding) == 100
    all_embeddings.append(embedding)
    all_words.append(word)
    word2id[word] = i
all_embeddings = np.array(all_embeddings)
while 1:
    context = input('Context: ')
    context_ids = []
    try:
        words = context.split(' ')
        for word in words:
            context_id = word2id[word]
            context_ids.append(context_id)
    except:
        print('Cannot find some words')
        continue
    context_embs = []
    for context_id in context_ids:
        context_emb = all_embeddings[context_id:context_id + 1]
        context_embs.append(context_emb)
    context_emb_sum = np.sum(context_embs, axis=0)
    d = cosine_similarity(context_emb_sum, all_embeddings)[0]
    d = zip(all_words, d)
    d = sorted(d, key=lambda x: x[1], reverse=True)
    for w in d[:3]:
        print(w)
