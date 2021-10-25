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
    pool = Pool(1)
    similarity_matrix = Manager().list()
    for i in range(len(keyword_count_by_essay[:1547])):
        pool.apply_async(cal_sim_matrix, args=(i, keyword_count_by_essay, similarity_matrix))
        # print(f"\r{i + 1}/{len(keyword_count_by_essay)}, complete!", end='')
    pool.close()
    pool.join()
    similarity_matrix = list(similarity_matrix)
    end_time = time.time()
    print("\nRunning time: ", end_time - start_time)
    print()

    print("Writing to csv...")
    start_time = time.time()
    with open('similarity_matrix.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        for i in similarity_matrix:
            f_csv.writerow(i)
    # result_file = open("similarity_matrix.txt", 'w')
    # for line in similarity_matrix:
    #     for i in line:
    #         result_file.write(str(i) + ' ')
    #     result_file.write('\n')
    # result_file.close()
    end_time = time.time()
    print("Running time: ", end_time - start_time)
