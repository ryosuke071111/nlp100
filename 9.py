#80
import bz2
import random
from collections import Counter
import pickle
import math
import pickle
from collections import OrderedDict
from scipy import sparse,io
from scipy import sparse,io
import sklearn.decomposition
import numpy as np

fname_counter_tc = 'counter_tc'
fname_counter_t = 'counter_t'
fname_counter_c = 'counter_c'
original = 'enwiki-20150112-400-r100-10576.txt.bz2'      # 1/100版^^;
fname_input = 'corpus80.txt'
fname_output = 'corpus81.txt'
fname_countries="countries.txt"
countries_out="countries_out.txt"
context_out="context.txt"
fname_dict_index_t = 'dict_index_t'
fname_matrix_x300 = 'matrix_x300'

#80
# with bz2.open(original,'rt') as data_file, open(corpus_in,mode='wt') as out_file :
#   for line in data_file:
#     tokens=[]
#     for chunk in line.split(' '):
#       token=chunk.strip('.,!?;:()[]\'"')
#       if len(token)>0:
#         tokens.append(token)
#     print(*tokens,sep=" ",end='\n', file=out_file)

#81
set_country = set()
dict_country = {}
with open(fname_countries, 'rt') as data_file:
    for line in data_file:
        words = line.split(' ')
        if len(words) > 1:

            # 集合に追加
            set_country.add(line.strip())

            # 辞書に追加
            if words[0] in dict_country:
                lengths = dict_country[words[0]]
                if not len(words) in lengths:
                    lengths.append(len(words))
                    lengths.sort(reverse=True)
            else:
                dict_country[words[0]] = [len(words)]

# 1行ずつ処理
with open(fname_input, 'rt') as data_file, \
        open(fname_output, mode='wt') as out_file:
    for line in data_file:

        # 1語ずつチェック
        tokens = line.strip().split(' ')
        result = []     # 結果のトークン配列
        skip = 0        # >0なら複数語の続き
        for i in range(len(tokens)):

            # 複数語の続きの場合はスキップ
            if skip > 0:
                skip -= 1
                continue

            # 1語目が辞書にある？
            if tokens[i] in dict_country:

                # 後続の語数を切り取って集合にあるかチェック
                hit = False
                for length in dict_country[tokens[i]]:
                    if ' '.join(tokens[i:i + length]) in set_country:

                        # 複数語の国を発見したので'_'で連結して結果に追加
                        result.append('_'.join(tokens[i:i + length]))
                        skip = length - 1       # 残りの語はスキップ
                        hit = True
                        break
                if hit:
                    continue

            # 複数語の国ではないので、そのまま結果に追加
            result.append(tokens[i])

        # 出力
        print(*result, sep=' ', end='\n', file=out_file)

#82
# context_dic={}
# with open(corpus_out,'rt') as data_file, open(context_out,"wt") as out_file:
#   for line in data_file:
#     line=line.strip().split()
#     for i in range(len(line)):
#       idx=random.randint(1,5)
#       try:
#         if line[i-idx] and line[i+idx]:
#           context_dic[line[i]]=line[i-idx:i]+line[i+1:i+idx]
#           for word in range(len(context_dic[line[i]])):
#             print("{}\t{}".format(line[i],context_dic[line[i]][word]),file=out_file)
#       except:
#         pass

# #83
# counter_tc = Counter()
# counter_t = Counter()
# counter_c = Counter()

# # 1行ずつ処理
# # work_tc = []
# # work_t = []
# # work_c = []

# # with open(context_out,'rt') as data_file:
# #   for i, line in enumerate(data_file,start=1):
# #     line=line.strip()
# #     tokens=line.split('\t')

# #     work_tc.append(line)
# #     work_t.append(tokens[0])
# #     work_c.append(tokens[1])

# #     if i % 1000000 == 0:
# #         counter_tc.update(work_tc)
# #         counter_t.update(work_t)
# #         counter_c.update(work_c)
# #         work_tc = []
# #         work_t = []
# #         work_c = []
# #         print('{} done.'.format(i))
# # counter_tc.update(work_tc)
# # counter_t.update(work_t)
# # counter_c.update(work_c)
# # with open(fname_counter_tc, 'wb') as data_file:
# #     pickle.dump(counter_tc, data_file)
# # with open(fname_counter_t, 'wb') as data_file:
# #     pickle.dump(counter_t, data_file)
# # with open(fname_counter_c, 'wb') as data_file:
# #     pickle.dump(counter_c, data_file)

# # print("N=",i)

# #84
# # fname_counter_tc = 'counter_tc'
# # fname_counter_t = 'counter_t'
# # fname_counter_c = 'counter_c'
# # fname_matrix_x = 'matrix_x'
# # fname_dict_index_t = 'dict_index_t'
# # N= 53403703        # 問題83で求めた定数

# # # Counter読み込み
# # with open(fname_counter_tc, 'rb') as data_file:
# #     counter_tc = pickle.load(data_file)
# # with open(fname_counter_t, 'rb') as data_file:
# #     counter_t = pickle.load(data_file)
# # with open(fname_counter_c, 'rb') as data_file:
# #     counter_c = pickle.load(data_file)

# # # {単語, インデックス}の辞書作成
# # dict_index_t = OrderedDict((key, i) for i, key in enumerate(counter_t.keys()))
# # dict_index_c = OrderedDict((key, i) for i, key in enumerate(counter_c.keys()))

# # # 行列作成
# # size_t = len(dict_index_t)
# # size_c = len(dict_index_c)
# # matrix_x = sparse.lil_matrix((size_t, size_c))

# # # f(t, c)を列挙して処理
# # for k, f_tc in counter_tc.items():
# #     if f_tc >= 10:
# #         tokens = k.split('\t')
# #         t = tokens[0]
# #         c = tokens[1]
# #         ppmi = max(math.log((N * f_tc) / (counter_t[t] * counter_c[c])), 0)
# #         matrix_x[dict_index_t[t], dict_index_c[c]] = ppmi

# # # 結果の書き出し
# # io.savemat(fname_matrix_x, {'matrix_x': matrix_x})
# # with open(fname_dict_index_t, 'wb') as data_file:
# #     pickle.dump(dict_index_t, data_file)

# # #85

# # fname_matrix_x = 'matrix_x'
# # fname_matrix_x300 = 'matrix_x300'

# # matrix_x=io.loadmat(fname_matrix_x)['matrix_x']
# # clf=sklearn.decomposition.TruncatedSVD(300)
# # matrix_x300 = clf.fit_transform(matrix_x)
# # io.savemat(fname_matrix_x300, {'matrix_x300': matrix_x300})

# # #86
# # fname_dict_index_t = 'dict_index_t'
# # fname_matrix_x300 = 'matrix_x300'

# # # 辞書読み込み
# # with open(fname_dict_index_t, 'rb') as data_file:
# #     dict_index_t = pickle.load(data_file)
# # matrix_x300=io.loadmat(fname_matrix_x300)['matrix_x300']
# # print(matrix_x300[dict_index_t['United_States']])

# #87
# def cos_sim(vec_a,vec_b):
#   norm_ab=np.linalg.norm(vec_a)*np.linalg.norm(vec_b)
#   if norm_ab!=0:
#     return np.dot(vec_a,vec_b)/norm_ab
#   else:
#     return -1
# # 辞書読み込み
# with open(fname_dict_index_t, 'rb') as data_file:
#     dict_index_t = pickle.load(data_file)

# # 行列読み込み
# # matrix_x300 = io.loadmat(fname_matrix_x300)['matrix_x300']

# # vec_a= matrix_x300[dict_index_t["United_States"]]
# # vec_b= matrix_x300[dict_index_t["U.S"]]

# # print(cos_sim(vec_a,vec_b))

# # #88
# # with open(fname_dict_index_t,'rb') as data_file:
# #   dict_index_t=pickle.load(data_file)
# #   print("dict_index_t:",dict_index_t)
# # matrix_x300=io.loadmat(fname_matrix_x300)['matrix_x300']

# # vec_England=matrix_x300[dict_index_t["England"]]
# # distances=[cos_sim(vec_England,matrix_x300[i]) for i in range(len(dict_index_t))]

# # index_sorted=np.argsort(distances)
# # keys=list(dict_index_t.keys())
# # for index in index_sorted[-2:-12:-1]:
# #   print('{}\t{}'.format(keys[index],distances[index]))


# #89
# matrix_x300=io.loadmat(fname_matrix_x300)['matrix_x300']

# vec=matrix_x300[dict_index_t['Spain']]-matrix_x300[dict_index_t['Madrid']]+matrix_x300[dict_index_t['Athens']]
# distances=[cos_sim(vec,matrix_x300[i]) for i in range(len(dict_index_t))]

# index_sorted=np.argsort(distances)
# keys=list(dict_index_t.keys())
# for index in index_sorted[:-11:-1]:
#   print('{}\t{}'.format(keys[index],distances[index]))



