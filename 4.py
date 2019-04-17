#形態素解析
#30
import MeCab
from collections import Counter
import matplotlib.pyplot as plt
fname="neko.txt"
fname_parsed="neko.txt.mecab"

def parse_neko():
  with open(fname) as data_file,\
      open(fname_parsed,mode="w") as out_file:
    mecab=MeCab.Tagger()
    out_file.write(mecab.parse(data_file.read()))

def neco_lines():
  with open(fname_parsed) as file_parsed:
    morphemes=[]
    for line  in file_parsed:
      cols=line.split('\t')
      if len(cols)<2:
        raise StopIteration
      res_cols=cols[1].split(",")
      morpheme={
        "surface":cols[0],
        "base":res_cols[6],
        "pos":res_cols[0],
        "pos1":res_cols[1]
      }
      morphemes.append(morpheme)
      if res_cols[1]=="句点":
        yield morphemes
        morphemes=[]

parse_neko()
neco_lines()
# lines=neco_lines()
# for line in lines:
#   print(line)

# 31
# verbs=set()
# verbs_test=[]
# lines=neco_lines()
# for line in lines:
#   for morpheme in line:
#     if morpheme["pos"]=="動詞":
#       verbs.add(morpheme["surface"])
#       verbs_test.append(morpheme['surface'])
# print(sorted(verbs,key=verbs_test.index))

#32
# verbs=set()
# lines=neco_lines()
# for line in lines:
#   for morpheme in line:
#     if morpheme["pos"]=="動詞":
#       verbs.add(morpheme["base"])
# print(sorted(verbs))

#33
# nouns=[]
# lines=neco_lines()
# cnt=0
# for line in lines:
#   for morpheme in line:
#     if morpheme["pos"]=="名詞" and morpheme["pos1"]=="サ変接続":
#       nouns.append(morpheme)
#       cnt+=1
#       # print(morpheme)
#     if cnt==5:
#       for noun in nouns:
#         print(noun)
#       exit()
# print(nouns)

#34
# lines=neco_lines()
# list_a_no_b=[]
# for line in lines:
#   if len(line)>2:
#     for i in range(1,len(line)-1):
#       if line[i]["surface"]=="の"\
#         and line[i-1]["pos"]=="名詞"\
#         and line[i+1]["pos"]=="名詞":
#         list_a_no_b.append(line[i-1]["surface"]+"の"+line[i+1]["surface"])
# a_no_b=set(list_a_no_b)
# print(a_no_b)

#35
# list_series_noun=[]
# for line in neco_lines():
#   nouns=[]
#   for morpheme in line:
#     if morpheme["pos"]=="名詞":
#       nouns.append(morpheme["surface"])
#     else:
#       if len(nouns)>1:
#         list_series_noun.append(''.join(nouns))
#       nouns=[]
#   if len(nouns)>1:
#     list_series_noun.append(''.join(nouns))
# series_noun=list_series_noun
# for series in sorted(series_noun,key=lambda x:len(x)):
#   print(series)

#36
# counter=Counter()
# for line in neco_lines():
#   counter.update([morpheme["surface"] for morpheme in line])
# print(counter.most_common()[:3])

#37
# counter=Counter()
# for line in neco_lines():
#   counter.update([morpheme["surface"] for morpheme in line])
# data=counter.most_common()[:10]
# data0=list(map(lambda x:x[0],data))
# data1=list(map(lambda x:x[1],data))
# ##グラフ設定
# # fp = FontProperties(
#     # fname='/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf')
# plt.bar(data0,data1,align="center")
# plt.show()

#38
# counter=Counter()
# for line in neco_lines():
#   counter.update([morpheme["surface"] for morpheme in line])
# data=counter.most_common()
# data0=list(map(lambda x:x[0],data))
# data1=list(map(lambda x:x[1],data))
# # plt.hist(data1,bins=20,range=(1,20))
# # plt.show()

# #39
# plt.scatter(range(1,len(data1)+1),data1)
# plt.xlim(1, len(data1) + 1)
# plt.ylim(1, data1[0])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

#係り受け解析
#英語テキスト処理
#データベース
#機械学習
#ベクトル空間方
#ベクトル空間方
