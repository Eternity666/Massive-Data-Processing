from gensim.models import word2vec
import pandas as pd
import numpy as np
import logging
import jieba
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = pd.read_csv("train_file.txt", sep="\t", encoding='utf8', header=None)
print(data.head())
sentence = list(data[1])

# # 过滤出中文及字符串以外的其他符号
# r = re.compile(r"[][【】\s+.!/_,$%^*(\"\']+|[+—！；「」》:：“”·‘’《，。？、~@#￥%…&*（）()]")
# for i in range(len(sentence)):
#     sentence[i] = r.sub('', sentence[i])


# 创建停用词列表
def stopwords_list():
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='ansi').readlines()]
    return stopwords


# 对文档进行中文分词
def seg_depart(doc):
    # print("正在分词")
    doc_depart = jieba.cut(doc.strip())
    # 创建一个停用词列表
    stopwords = stopwords_list()
    # 输出结果为outstr
    words = []
    # 去停用词
    for word in doc_depart:
        if word not in stopwords:
            if word != ' ':
                words.append(word)

    return words


sentence_words = []
for i in range(len(sentence)):
    if type(sentence[i]) is not float:
        sentence_words.append(seg_depart(sentence[i]))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(sentence_words, min_count=10, sg=0, epochs=20)
model.save("word2vec.model")


# 测试
for i in model.wv.most_similar(u"期货"):
    print(i[0], i[1])

# words = []
# for i in sentence_words:
#     words += i
# words = list(set(words))

# np.random.shuffle(words)
# random_words = words[:100]
# X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(model.wv[random_words])
#
# plt.figure()
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
# for i in range(len(X_tsne)):
#     x = X_tsne[i][0]
#     y = X_tsne[i][1]
#     plt.text(x, y, random_words[i], size=16)
#
# plt.show()
