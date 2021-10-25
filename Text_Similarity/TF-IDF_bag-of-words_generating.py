import time
import csv
from math import *
from multiprocessing import Pool, cpu_count, Manager

# file_path = "199801_clear (1).txt"
# file = open(file_path)
# all_essay = file.read()
# file.close()
# split_essay = all_essay.split('\n\n')
# print(len(split_essay))


file_path = "199801_clear (1).txt"
file = open(file_path, encoding='gbk')
file_lines = file.readlines()
file.close()
# print(file_lines[16] == '')

stop_words_path = "stopwords_1.txt"
stop_words_file = open(stop_words_path)
stop_words = stop_words_file.read()
stop_words = stop_words.split('\n')

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

# print(len(essay_split))
# print(essay_split[-1])

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

essay_count = len(words_list_by_essay)
print(essay_count)
print(len(words_list_by_essay))
# for i in words_list_by_essay:
#     if len(i) < 10:
#         print(i)

# 计算每篇文章各自的总词数
all_words_count_by_essay = []
for i in words_list_by_essay:
    all_words_count_by_essay.append(len(i))

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

index = [i for i in range(len(all_words_list))]
word2Index = dict(zip(all_words_list, index))
Index2Word = dict(zip(index, all_words_list))

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

# print(keywords_list_by_essay)

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

# print("Writing to txt...")
# start_time = time.time()
# result_file = open("bag-of-words.txt", 'w')
# for line in keyword_count_by_essay:
#     for i in line:
#         result_file.write(str(i) + ' ')
#     result_file.write('\n')
# result_file.close()
# end_time = time.time()
# print("Running time: ", end_time - start_time)

print("Writing to csv...")
start_time = time.time()
with open('bag-of-words.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    for i in keyword_count_by_essay:
        f_csv.writerow(i)
end_time = time.time()
print("Running time: ", end_time - start_time)
