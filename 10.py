import pickle
from collections import OrderedDict
import numpy as np
from scipy import io
import word2vec

#90
fname_input = 'corpus81.txt'
fname_word2vec_out = 'vectors.txt'
fname_dict_index_t = 'dict_index_t'
fname_matrix_x300 = 'matrix_x300'
fname_output = 'family_out.txt'

word2vec.word2vec(train=fname_input, output=fname_word2vec_out,
    size=300, threads=4, binary=0)

# # その結果を読み込んで行列と辞書作成
# with open(fname_word2vec_out, 'rt') as data_file:

#     # 先頭行から用語数と次元を取得
#     work = data_file.readline().split(' ')
#     size_dict = int(work[0])
#     size_x = int(work[1])

#     # 辞書と行列作成
#     dict_index_t = OrderedDict()
#     matrix_x = np.zeros([size_dict, size_x], dtype=np.float64)

#     for i, line in enumerate(data_file):
#         work = line.strip().split(' ')
#         try:
#           dict_index_t[work[0]] = i
#           matrix_x[i] = work[:]
#         except:
#           pass

# # 結果の書き出し
# io.savemat(fname_matrix_x300, {'matrix_x300': matrix_x})
# with open(fname_dict_index_t, 'wb') as data_file:
#     pickle.dump(dict_index_t, data_file)

# # fname_dict_index_t = 'dict_index_t'
# # fname_matrix_x300 = 'matrix_x300'

# # 辞書読み込み
# with open(fname_dict_index_t, 'rb') as data_file:
#     dict_index_t = pickle.load(data_file)
# matrix_x300=io.loadmat(fname_matrix_x300)['matrix_x300']
# print(matrix_x300[dict_index_t['United_States']])


#91
fname_input = 'questions-words.txt'
fname_output = 'family.txt'

with open(fname_input, 'rt') as data_file, \
        open(fname_output, 'wt') as out_file:

    target = False      # 対象のデータ？
    for line in data_file:

        if target is True:

            # 対象データの場合は別のセクションになるまで出力
            if line.startswith(': '):
                break
            print(line.strip(), file=out_file)

        elif line.startswith(': family'):

            # 対象データ発見
            target = True

#92
import pickle
from collections import OrderedDict
from scipy import io
import numpy as np

fname_dict_index_t = 'dict_index_t'
fname_matrix_x300 = 'matrix_x300'
fname_input = 'family.txt'
fname_output = 'family_out.txt'


# def cos_sim(vec_a, vec_b):
#     '''コサイン類似度の計算
#     ベクトルvec_a、vec_bのコサイン類似度を求める

#     戻り値：
#     コサイン類似度
#     '''
#     norm_ab = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
#     if norm_ab != 0:
#         return np.dot(vec_a, vec_b) / norm_ab
#     else:
#         # ベクトルのノルムが0だと似ているかどうかの判断すらできないので最低値
#         return -1


# # 辞書読み込み
# with open(fname_dict_index_t, 'rb') as data_file:
#         dict_index_t = pickle.load(data_file)
# keys = list(dict_index_t.keys())

# # 行列読み込み
# matrix_x300 = io.loadmat(fname_matrix_x300)['matrix_x300']

# # 評価データ読み込み
# with open(fname_input, 'rt') as data_file, \
#         open(fname_output, 'wt') as out_file:

#     for line in data_file:
#         cols = line.split(' ')

#         try:

#             # ベクトル計算
#             vec = matrix_x300[dict_index_t[cols[1]]] \
#                     - matrix_x300[dict_index_t[cols[0]]] \
#                     + matrix_x300[dict_index_t[cols[2]]]

#             # コサイン類似度の一番高い単語を抽出
#             dist_max = -1
#             index_max = 0
#             result = ''
#             for i in range(len(dict_index_t)):
#                 dist = cos_sim(vec, matrix_x300[i])
#                 if dist > dist_max:
#                     index_max = i
#                     dist_max = dist

#             result = keys[index_max]

#         except KeyError:

#             # 単語がなければ0文字をコサイン類似度-1で出力
#             result = ''
#             dist_max = -1

#         # 出力
#         print('{} {} {}'.format(line.strip(), result, dist), file=out_file)
#         print('{} {} {}'.format(line.strip(), result, dist))

# #93
# fname_input="family_out.txt"
# with open(fname_input,'rt') as data_file:
#   correct=0
#   total=0

#   for line in data_file:
#     cols=line.split()
#     total+=1
#     if cols[3]==cols[4]:
#       correct+=1
# print('{}({}/{})'.format(correct/total,correct,total))

#94

#96
# import pickle
# from collections import OrderedDict
# from scipy import io
# import numpy as np

# fname_dict_index_t = 'dict_index_t'
# fname_matrix_x300 = 'matrix_x300'
# fname_countries = 'countries.txt'

# fname_dict_new = 'dict_index_country'
# fname_matrix_new = 'matrix_x300_country'


# # 辞書読み込み
# with open(fname_dict_index_t, 'rb') as data_file:
#         dict_index_t = pickle.load(data_file)

# # 行列読み込み
# matrix_x300 = io.loadmat(fname_matrix_x300)['matrix_x300']

# # 辞書にある用語のみの行列を作成
# dict_new = OrderedDict()
# matrix_new = np.empty([0, 300], dtype=np.float64)
# count = 0

# with open(fname_countries, 'rt') as data_file:
#     for line in data_file:
#         try:
#             word = line.strip().replace(' ', '_')
#             index = dict_index_t[word]
#             matrix_new = np.vstack([matrix_new, matrix_x300[index]])
#             dict_new[word] = count
#             count += 1
#         except:
#             pass

# # 結果の書き出し
# io.savemat(fname_matrix_new, {'matrix_x300': matrix_new})
# with open(fname_dict_new, 'wb') as data_file:
#     pickle.dump(dict_new, data_file)


#97
# import pickle
# from collections import OrderedDict
# from scipy import io
# import numpy as np
# from sklearn.cluster import KMeans

# fname_dict_index_t = 'dict_index_country'
# fname_matrix_x300 = 'matrix_x300_country'

# # 辞書読み込み
# with open(fname_dict_index_t, 'rb') as data_file:
#         dict_index_t = pickle.load(data_file)

# # 行列読み込み
# matrix_x300 = io.loadmat(fname_matrix_x300)['matrix_x300']

# # KMeansクラスタリング
# predicts = KMeans(n_clusters=5).fit_predict(matrix_x300)

# # (国,分類番号)のリスト作成
# result = zip(dict_index_t.keys(), predicts)

# # 分類番号でソートして表示
# for country, category in sorted(result, key=lambda x: x[1]):
#     print('{}\t{}'.format(category, country))

#98
# import pickle
# from collections import OrderedDict
# from scipy import io
# import numpy as np

# from scipy.cluster.hierarchy import ward, dendrogram
# from matplotlib import pyplot as plt

# fname_dict_index_t = 'dict_index_country'
# fname_matrix_x300 = 'matrix_x300_country'


# # 辞書読み込み
# with open(fname_dict_index_t, 'rb') as data_file:
#         dict_index_t = pickle.load(data_file)

# # 行列読み込み
# matrix_x300 = io.loadmat(fname_matrix_x300)['matrix_x300']

# # Ward法でクラスタリング
# ward = ward(matrix_x300)
# print(ward)

# # デンドログラム表示
# dendrogram(ward, labels=list(dict_index_t.keys()), leaf_font_size=8)
# plt.show()

#99
import pickle
from collections import OrderedDict
from scipy import io
import numpy as np

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

fname_dict_index_t = 'dict_index_country'
fname_matrix_x300 = 'matrix_x300_country'


# 辞書読み込み
with open(fname_dict_index_t, 'rb') as data_file:
        dict_index_t = pickle.load(data_file)

# 行列読み込み
matrix_x300 = io.loadmat(fname_matrix_x300)['matrix_x300']

# t-SNE
t_sne = TSNE(perplexity=30, learning_rate=500).fit_transform(matrix_x300)
print(t_sne)

# KMeansクラスタリング
predicts = KMeans(n_clusters=5).fit_predict(matrix_x300)

# 表示
fig, ax = plt.subplots()
cmap = plt.get_cmap('Set1')
for index, label in enumerate(dict_index_t.keys()):
    cval = cmap(predicts[index] / 4)
    ax.scatter(t_sne[index, 0], t_sne[index, 1], marker='.', color=cval)
    ax.annotate(label, xy=(t_sne[index, 0], t_sne[index, 1]), color=cval)
plt.show()
