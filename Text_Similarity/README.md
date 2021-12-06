# 文本间两两相似度计算

> **2101210676 杨智昊**



## 本文摘要

- 本文首先对题目所给公式的合理性进行探讨，认为该公式不适合计算文本之间两两相似度的应用场景，并给出了原因；
- 对题目所给语料进行预处理，总计得到3148篇文本，将同文章的词合并在一个列表中，并且去掉中文停用词；
- 基线模型讨论与算法优化，通过理论分析和实验确定更优的解法；
- 计算每篇文章中每个词的TF-IDF值，并降序排列，每篇筛选出前10（也可以是其他数字，可根据实际情况和计算性能进行调整）个TF-IDF较大的词作为关键词；
- 将每篇文章的关键词组合在一起求并集，组成关键词词袋，并以该词袋为基准，统计每篇文章在该词袋上的词频，生成可以代表每篇文章的向量；
- 对向量计算两两相似度（可选的计算方法有余弦相似度、欧氏距离、Jacard相似度和海明距离等等），生成相似度矩阵（用下三角矩阵存储）；
- 由于数据量较大且个人PC算力有限，基线模型的运行时间非常长，所以本文考虑使用多进程优化来提高运算效率。
- 完成本次作业的数据和全部代码，已共享在Github平台（[链接](https://github.com/Eternity666/Massive-Data-Processing/tree/main/Text_Similarity)）上，欢迎大家提出建议~~

> **说明**：本次作业的代码中，数据全部用Python自带的数据结构（列表、字典和元组等）进行存储，计算方法全部基于自带数据结构进行编写，没有调用numpy、pandas、sklearn等任何库。



## 目录

[TOC]



## 思考

- 看起来进行了很多适应性调整的常用公式，是不是适合这题目的要求？
- 如果适合，请选用，如果不适合，请依据本节课内容自行拟定

<div align=center>
<img src="https://github.com/Eternity666/Massive-Data-Processing/blob/main/Text_Similarity/img/image-20211018195847138.png"/>
</div>

- 我认为**不适合**，原因如下
  - 使用该公式计算文本相似度时，必须指定其中一个文本为查询语句```Query```，其余文本为文档语句```Document```，并进行长度规整。
  - 本题目要求计算文本两两之间的相似度，那么对于每一对文本，理论上会计算得到两个不同的相似度（即两个文本分别作为查询语句时所得到的相似度）。
  - 用不同文本作为查询语句时，文档语句会相应不同，进而平均长度也是不一样的，因而所做长度规整的标准就失去了统一性。
  - 因此，用该公式所计算的两两相似度不是在统一标准下得到的，相互之间无法做比较，所以不适合。



## 语料预处理

- 首先读入语料文件和停用词文件

```python
file_path = "199801_clear (1).txt"
file = open(file_path, encoding='gbk')
file_lines = file.readlines()
file.close()
# print(file_lines[16] == '')

stop_words_path = "stopwords_1.txt"
stop_words_file = open(stop_words_path)
stop_words = stop_words_file.read()
stop_words = stop_words.split('\n')
```



- 由原始数据可以看到，每一行的第一个元素记录了文章的标号。```XXXXXXXX-XX-XXX```可作为每篇文章的特定标识，后面的```-XXX```为每篇文章的行号。因此，我们以前者为依据划分文章。

<div align=center>
<img src="https://github.com/Eternity666/Massive-Data-Processing/blob/main/Text_Similarity/img/image-20211024123409578.png"/>
</div>



- 可以注意到，在原始数据中，不同文章之间大部分会以一个空行隔开。但也会出现如下图两种例外情况：

  - 不同文章之间没有空行；

<div align=center>
<img src="https://github.com/Eternity666/Massive-Data-Processing/blob/main/Text_Similarity/img/image-20211024123751078.png"/>
</div>

  - 同一篇文章内，会出现空行分隔。

<div align=center>
<img src="https://github.com/Eternity666/Massive-Data-Processing/blob/main/Text_Similarity/img/image-20211024123850978.png" alt="image-20211024123850978" style="zoom: 67%;" />
</div>
  
​		因此，不能简单使用```\n```来划分文章。本作业编写的划分文章代码如下，得到的文本数量为**3148**篇。





```python
pre_title = file_lines[0][:15]
essay_split = []
now_essay = ""

for i in file_lines:
    if i == '\n':
        continue
    now_title = i[:15]
    if now_title == pre_title:
        now_essay += i
    else:
        essay_split.append(now_essay)
        now_essay = i
        pre_title = now_title
essay_split.append(now_essay)
```



- 将每个词语的后面的词性去掉，并且去掉中文停用词，得到每篇文章的词语列表，储存在```words_list_by_essay```列表中。

```python
words_list_by_essay = []
for i in essay_split:
    words = i.split()
    words_list = []
    for j in words:
        if j[:6] == '199801':
            continue
        j_new = j.split('/')[0]
        if j_new in stop_words:
            continue
        words_list.append(j_new)
    words_list_by_essay.append(words_list)
```



## 基线模型实现与讨论

### 基线模型1

- 将所有文本中出现的全部词语构成一个词袋，词语数量为54074，统计每篇文章中词语在这个“巨型”词袋上的词频，构成文章的向量表示。

```python
# 统计全部词的数量
print("\nBuilding a list including all words...")
count = 0
all_words_list = []
for i in range(len(words_list_by_essay)):
    for j in words_list_by_essay[i]:
        if j not in all_words_list:
            all_words_list.append(j)
    print(f"\r{i + 1}/3148, complete!", end='')
print("\nNumber of all words: ", len(all_words_list))

print("\nBuilding Bag-of-words Model...")
start_time = time.time()
word_count_by_essay = []
for i in range(len(words_list_by_essay)):
    word_count = [0 for _ in range(len(all_words_list))]
    for j in words_list_by_essay[i]:
        word_count[word2Index[j]] += 1
    word_count_by_essay.append(word_count)
    print(f"\r{i + 1}/3148, complete!", end='')
end_time = time.time()
print("\nRunning time: ", end_time - start_time)
```

- 以此向量表示方法为基准，计算文本的两两相似度（此处以余弦相似度为例）。

```python
def cal_cos_similarity(vec1, vec2):
    inner_product = 0
    norm1 = 0
    norm2 = 0
    for i in range(len(vec1)):
        inner_product += (vec1[i] * vec2[i])
        norm1 += vec1[i] ** 2
        norm2 += vec2[i] ** 2
    if norm1 == 0 or norm2 == 0:
        return 0
    norm1 = norm1 ** 0.5
    norm2 = norm2 ** 0.5
    return round(inner_product / (norm1 * norm2), 5)


# 计算相似度矩阵（下三角）
print("\nBuilding similarity matrix...")
start_time = time.time()
similarity_matrix = []
for i in range(len(word_count_by_essay)):
    similarity_vector = []
    for j in range(i+1):
        if i == j:
            similarity_vector.append(1.0)
        else:
     similarity_vector.append(cal_cos_similarity(word_count_by_essay[i], word_count_by_essay[j]))
    similarity_matrix.append(similarity_vector)
    print(f"\r{i + 1}/3148, complete!", end='')
end_time = time.time()
print("\nRunning time: ", end_time - start_time)
print()
```

- 程序每一步的运行时间统计如下表

|     计算模块     |    运行时间    |
| :--------------: | :------------: |
| 统计全部词的数量 |    3 m 33 s    |
|   建立词袋模型   |    11.87 s     |
|  计算两两相似度  | 预计 83 h 38 m |

- 其中，计算两两相似度模块的运行时间由前几次循环的运行时间估算得到。



### 基线模型2

- 对每篇文章的每个词语计算其TF-IDF值，构成每篇文章的向量表示。

```python
print("\nBuilding TF-IDF matrix...")
start_time = time.time()
tf_idf_matrix = []
for i in range(len(word_count_by_essay)):
    start = time.time()
    tf_idf_vec = []
    for j in range(len(word_count_by_essay[0])):
        tf = word_count_by_essay[i][j] / sum(word_count_by_essay[i])
        count_including_the_word = 0
        for k in range(len(word_count_by_essay)):
            if word_count_by_essay[k][j] > 0:
                count_including_the_word += 1
        idf = log(len(words_list_by_essay) / (count_including_the_word + 1))
        tf_idf_vec.append(tf * idf)
    tf_idf_matrix.append(tf_idf_vec)
    end = time.time()
    print(f"\r{i + 1}/3148, complete! Running time: {end - start}s.", end='')

end_time = time.time()
print("\nRunning time: ", end_time - start_time)
```

- 根据已进行的几轮循环，可以估算该代码块的运行时间为612 h 28 m。



### 基线模型讨论

- 显然，上述两个基线模型的运行时间过长，普通电脑难以招架。
- 另外，基线模型也存在着一些冗余计算。例如，在整个语料库中，每个词语的IDF值应该是相同的，所以不需要对每个文章中的每个词语都计算一次IDF，这极大提高了时间开销。
- 以语料中所有的词构成词袋，会有许多噪声参与计算，而且计算量也非常大。因此，我们有必要对词语进行特征选择，筛选出每篇文章的关键词构成新的词袋。



## 高效率编程挑战

### 算法优化

> TF-IDF（Term Frequency-inverse Document Frequency）是一种针对关键词的统计分析方法，用于评估一个词对一个文件集或者一个语料库的重要程度。一个词的重要程度跟它在文章中出现的次数成正比，跟它在语料库出现的次数成反比。这种计算方式能有效避免常用词对关键词的影响，提高了关键词与文章之间的相关性。[1]

- TF=（某词在文档中出现的次数/文档的总词量）
- IDF=log（语料库中文档总数/（包含该词的文档数+1））
- TF-IDF = TF * IDF

本文使用TF-IDF算法来计算每个词语相对每篇文章的重要程度，对每篇文章筛选出前10（也可以是其他数字，可根据实际情况和计算性能进行调整，本文以10个为例）个TF-IDF值较大的词作为文章关键词。

此外，针对**基线模型2**中出现的IDF冗余计算的问题，我将所有词语的IDF值在前面先行计算，储存在```word2essay_count```字典中。

```python
# 计算idf中log中的分母
print("\nCalculating the denominator in idf...")
start_time = time.time()
essay_count = [0 for _ in range(len(all_words_list))]
word2essay_count = dict(zip(all_words_list, essay_count))
tmp_count = 0
for i in all_words_list:
    for j in words_list_by_essay:
        if i in j:
            word2essay_count[i] += 1
    tmp_count += 1
    print(f'\r{tmp_count}/{len(all_words_list)}, complete!', end='')
end_time = time.time()
print("\nRunning time: ", end_time - start_time)
print(word2essay_count)

# 对每篇文章，筛选出 NUM_keywords_to_select 个关键词
print('\nSelecting keywords in each essay...')
start_time = time.time()
keywords_list_by_essay = []
NUM_keywords_to_select = 10
for i in range(len(words_list_by_essay)):
    keywords_tf_idf = dict()
    keywords = []
    for j in words_list_by_essay[i]:
        tf = word_count_by_essay[i][word2Index[j]] / all_words_count_by_essay[i]
        idf = log(len(words_list_by_essay) / (word2essay_count[j] + 1))
        tf_idf = tf * idf
        keywords_tf_idf[j] = tf_idf
    keywords_tf_idf = sorted(keywords_tf_idf.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # 返回的是元组
    keywords_tf_idf = dict(keywords_tf_idf)
    keywords = list(keywords_tf_idf.keys())[:NUM_keywords_to_select]
    keywords_list_by_essay.append(keywords)
    print(f"\r{i + 1}/3148, complete!", end='')
end_time = time.time()
print("\nRunning time: ", end_time - start_time)
```

将每篇文章的关键词组合在一起求并集，组成关键词词袋（共15407个关键词），并以该词袋为基准，统计每篇文章在该词袋上的词频，生成可以代表每篇文章的向量。

```python
all_keywords = [i for i in keywords_list_by_essay[0]]
# print(all_keywords)
for i in range(1, len(keywords_list_by_essay)):
    for j in keywords_list_by_essay[i]:
        if j not in all_keywords:
            all_keywords.append(j)
    # print(all_keywords)
print("Number of all keywords: ", len(all_keywords))

index = [i for i in range(len(all_keywords))]
keyword2Index = dict(zip(all_keywords, index))
Index2Keyword = dict(zip(index, all_keywords))

print("\nBuilding a Bag-of-keywords Model...")
start_time = time.time()
keyword_count_by_essay = []
for i in range(len(words_list_by_essay)):
    keyword_count = [0 for _ in range(len(all_keywords))]
    for j in words_list_by_essay[i]:
        if j in all_keywords:
            keyword_count[keyword2Index[j]] += 1
    keyword_count_by_essay.append(keyword_count)
    print(f"\r{i + 1}/3148, complete!", end='')
end_time = time.time()
print("\nRunning time: ", end_time - start_time)
```

为了使相似度的计算更加准确，消除文本长度的影响，我们也可以用词袋中每个词语的TF-IDF值替代词频，作为每篇文本的向量表示。

```python
print("\nBuilding Bag-of-keywords Model by using TF-IDF...")
start_time = time.time()
keyword_count_by_essay = []
for i in range(len(words_list_by_essay)):
    keyword_count = [0 for _ in range(len(all_keywords))]
    for j in words_list_by_essay[i]:
        if j in all_keywords:
            keyword_count[keyword2Index[j]] += 1
    for j in range(len(keyword_count)):
        tf = keyword_count[j] / all_words_count_by_essay[i]
        idf = log(len(words_list_by_essay) / (word2essay_count[Index2Keyword[j]] + 1))
        tf_idf = tf * idf
    keyword_count_by_essay.append(keyword_count)
    print(f"\r{i + 1}/3148, complete!", end='')
end_time = time.time()
print("\nRunning time: ", end_time - start_time)
```

随后进行两两相似度计算

```python
def cal_cos_similarity(vec1, vec2):
    inner_product = 0
    norm1 = 0
    norm2 = 0
    for i in range(len(vec1)):
        inner_product += (vec1[i] * vec2[i])
        norm1 += vec1[i] ** 2
        norm2 += vec2[i] ** 2
    if norm1 == 0 or norm2 == 0:
        return 0
    norm1 = norm1 ** 0.5
    norm2 = norm2 ** 0.5
    return round(inner_product / (norm1 * norm2), 5)


# 计算相似度矩阵（下三角）
print("\nBuilding similarity matrix...")
start_time = time.time()
similarity_matrix = []
for i in range(len(word_count_by_essay)):
    start = time.time()
    similarity_vector = []
    for j in range(i+1):
        if i == j:
            similarity_vector.append(1.0)
        else:
            similarity_vector.append(cal_cos_similarity(word_count_by_essay[i], word_count_by_essay[j]))
    similarity_matrix.append(similarity_vector)
    end = time.time()
    print(f"\r{i + 1}/3148, complete! Running time: {end - start}", end='')
end_time = time.time()
print("\nRunning time: ", end_time - start_time)
print()
```

统计每个计算模块的运行时间如下表

|          计算模块          | 运行时间  |
| :------------------------: | :-------: |
| 计算IDF中log真数部分的分母 | 9 m 54 s  |
|    对每篇文章筛选关键词    |  22.81 s  |
|     建立关键词词袋模型     |  1.98 s   |
|       计算两两相似度       | 14 h 21 m |

- 可以看到，经过特征选择与计算方法优化的程序在计算两两相似度时的速度提高了很多，但程序主要的时间开销还是两两相似度的计算。

- 另外，每次执行程序时，CPU的利用率仅占30%左右，没有充分利用CPU的计算资源。经过查看计算机属性和程序运行时的进程ID（PID），发现本机CPU有四个核，但每次运行程序时，操作系统只会给程序分配一个进程ID，调用一个CPU内核。

- 所以，我们自然想到尝试多进程优化，充分利用多核CPU的计算资源。



### 多进程优化

#### 实验环境

- 处理器：Intel(R) Core(TM) i5-7300HQ CPU @ 2.50 GHz

- 内存：12 GB
- CPU核数：4
- 操作系统：Windows 10 家庭中文版

本文中上述所有实验与下述多进程优化实验也均在该环境中进行。



- 本机PC的CPU有四个核，如果单纯地编程序进行计算，上述实验只可以调用一个核，分配一个进程ID进行计算，无法充分利用计算资源，计算效率也就比较低。
- python中的多线程其实并不是真正的多线程，如果想要充分地使用多核CPU的资源，在python中大部分情况需要使用多进程。[2]

- 并且，计算相似度矩阵的每一行是互不影响的，因此不会因为并行计算而降低计算的准确度。
- 因此，通过并行编程可以充分利用多核CPU，提高运算效率，降低时间成本。



#### 相似度矩阵并行计算

代码展示如下，词袋模型预先以csv的形式存储在了本地中，有助于直接展示相似度计算代码。

```python
import time
import csv
import os
from math import *
from multiprocessing import Pool, cpu_count, Manager


def cal_cos_similarity(vec1, vec2):
    inner_product = 0
    norm1 = 0
    norm2 = 0
    for i in range(len(vec1)):
        inner_product += (vec1[i] * vec2[i])
        norm1 += vec1[i] ** 2
        norm2 += vec2[i] ** 2
    if norm1 == 0 or norm2 == 0:
        return 0
    norm1 = norm1 ** 0.5
    norm2 = norm2 ** 0.5
    return round(inner_product / (norm1 * norm2), 4)


# 计算相似度矩阵（下三角）
def cal_sim_matrix(i, keyword_count_by_essay, similarity_matrix):
    print('执行任务%s (%s)...' % (i, os.getpid()))
    start = time.time()
    similarity_vector = []
    for j in range(i + 1):
        if i == j:
            similarity_vector.append(1.0)
        else:
            similarity_vector.append(cal_cos_similarity(keyword_count_by_essay[i], keyword_count_by_essay[j]))
    similarity_matrix.append(similarity_vector)
    end = time.time()
    print('任务 %s 运行了 %0.2f seconds.' % (i, (end - start)))


if __name__ == '__main__':
    # 读入csv
    keyword_count_by_essay = []
    with open('bag-of-words.csv')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            for i in range(len(row)):
                row[i] = int(row[i])
            keyword_count_by_essay.append(row)

    print("\nBuilding similarity matrix...")
    start_time = time.time()
    num_cores = cpu_count()
    pool = Pool(num_cores)
    similarity_matrix = Manager().list()
    for i in range(len(keyword_count_by_essay)):
        pool.apply_async(cal_sim_matrix, args=(i, keyword_count_by_essay, similarity_matrix))
        # print(f"\r{i + 1}/{len(keyword_count_by_essay)}, complete!", end='')
    pool.close()
    pool.join()
    similarity_matrix = list(similarity_matrix)
    end_time = time.time()
    print("\nRunning time: ", end_time - start_time)
    print()

    print("Writing to txt...")
    start_time = time.time()
    result_file = open("similarity_matrix.txt", 'w')
    for line in similarity_matrix:
        for i in line:
            result_file.write(str(i) + ' ')
        result_file.write('\n')
    result_file.close()
    end_time = time.time()
    print("Running time: ", end_time - start_time)
    
```

使用全部4个核进行运算时，程序运行不完，电脑会自动关机。所以只好设置3个核参与并行计算，计算时间为 **5 h 4 m**，相对于单核串行，显著提高了程序的计算效率。

诸如此类数据量比较大的问题，将数据和代码部署到算力较大的服务器上计算更为合适。优点有二：

- 服务器CPU数较多，核数也较多，可以极大提高计算效率。
- 服务器可以将程序挂在后台运行，不会因为电脑关机而中断程序的执行。



## 结果展示

程序计算得到文本之间的两两相似度矩阵，将部分展示如下图

<div align=center>
<img src="https://github.com/Eternity666/Massive-Data-Processing/blob/main/Text_Similarity/img/image-20211025153724193.png"/>
</div>

同样的，给定两篇文章的编号，我们也可以通过运行查询代码直接获取两篇文章的相似度。



## 总结与讨论

- 本次作业不依赖任何库，仅使用Python自带的数据结构和计算方式完成数据预处理、基线模型的搭建和TF-IDF算法的实现，自己的编程能力得到提高。
- 本次作业数据量之大，自己的电脑难以招架基线模型，也激励我主动去寻找更优的算法优化和并行优化方法。
- 对于海量数据的处理问题，相对于串行，并行计算会发挥极大的作用。



## 参考资料

[1] https://zhuanlan.zhihu.com/p/94446764

[2] https://www.cnblogs.com/kaituorensheng/p/4445418.html
