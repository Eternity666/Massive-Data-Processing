import pandas as pd
from gensim.models import LdaModel
from gensim import corpora
import jieba

data = pd.read_csv('text.csv', header=None)
abstract_raw = data[4]
abstract_raw = list(abstract_raw)

# 加载自己的词典，避免一些关键词被分割
jieba.load_userdict('mydict.utf8')


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


abstract = []
for i in range(len(abstract_raw)):
    abstract.append(seg_depart(abstract_raw[i]))

# for i in abstract[:10]:
#     print(i)

train_text = abstract[:320]
dictionary = corpora.Dictionary(train_text)
dictionary.filter_n_most_frequent(200)
corpus = [dictionary.doc2bow(text) for text in train_text]

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=6)

topic_list = lda.print_topics(num_words=20)
for i in topic_list:
    print(i)
print()


# 获取一篇新文档的主题
new_text = abstract[-2]
doc_bow = dictionary.doc2bow(new_text)
doc_lda = lda[doc_bow]
print(abstract_raw[-2])
print(doc_lda)

