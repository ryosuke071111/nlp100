#70
import codecs
import random
import snowballstemmer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

eta=6.0
epoch=800

fname_pos = 'rt-polaritydata/rt-polarity.pos'
fname_neg = 'rt-polaritydata/rt-polarity.neg'
fname_smt = 'sentiment.txt'
fencoding = 'cp1252'        # Windows-1252らしい
fname_sentiment = 'sentiment.txt'
fname_features = 'features.txt'
fname_theta = 'theta.npy'
fname_result = 'result.txt'
fname_work = 'work.txt'

result=[]
with codecs.open(fname_pos,'r',fencoding) as file_pos:
  result.extend(['+1{}'.format(line.strip()) for line in file_pos])

with codecs.open(fname_neg,'r',fencoding) as file_neg:
  result.extend(['-1{}'.format(line.strip()) for line in file_neg])

random.shuffle(result)

with codecs.open(fname_smt,'w',fencoding) as file_out:
  print(*result,sep="\n",file=file_out)

cnt_pos=0
cnt_neg=0

with codecs.open(fname_smt,"r",fencoding) as file_out:
  for line in file_out:
    if line.startswith('+1'):
      cnt_pos+=1
    elif line.startswith('-1'):
      cnt_neg+=1

print('pos:{},neg:{}'.format(cnt_neg,cnt_neg))

#71
stop_words = (
    'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,'
    'as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,'
    'either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,'
    'him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,'
    'likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,'
    'on,only,or,other,our,own,rather,said,say,says,she,should,since,so,'
    'some,than,that,the,their,them,then,there,these,they,this,tis,to,too,'
    'twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,'
    'will,with,would,yet,you,your').lower().split(',')

def is_stopword(str):
  return str.lower() in stop_words
# tests=("a","your","often","ON","0","z","bout","\n","")
# for test in tests:
#   print (test,is_stopword(test))

#72
stemmer=snowballstemmer.stemmer('english')
word_counter=Counter()
with codecs.open(fname_sentiment,'r',fencoding) as file_in:
  for line in file_in:
    for word in line[3:].split(' '):
      word = word.strip()
      if is_stopword(word):
        continue
      word=stemmer.stemWord(word)

      if word != "!" and word!="?" and len(word)<=1:
        continue
      word_counter.update([word])
features=[word for word,count in word_counter.items() if count>=6]

with codecs.open(fname_features,'w',fencoding) as file_out:
  print(*features,sep="\n",file=file_out)

#73 素性抽出
def sigmoid(x,w):
  return 1.0/(1+np.exp(-x.dot(w)))

def cost(x,w,t):
  size=t.size
  y=sigmoid(x,w)
  cross_enthoropy= 1/size*np.sum(-t*np.log(y)-(1-t)*np.log(1-y))
  return cross_enthoropy

#コスト関数の微分
def gradient(x,w,t):
  size=t.size
  y=sigmoid(x,w)
  grad=1/size*(y-t).dot(x)
  return grad

#レビューごとに素性を疎ベクトルに入れていく。
def extract_features(data,dict_features):
  data_one_x=np.zeros(len(dict_features)+1)
  data_one_x[0]=1
  for word in data.split():
    word=word.strip()
    if is_stopword(word):
      continue
    word=stemmer.stemWord(word)
    try:
      data_one_x[dict_features[word]]=1
    except:
      pass
  return data_one_x

def load_dict_features():
  with codecs.open(fname_features,'r',fencoding) as file_in:
    return {line.strip():i for i,line in enumerate(file_in,start=1)}

#文章データから教師データ作成（1,0,0,1のスパースベクトルと行頭の+1/-1からラベル作成）
def create_training_set(sentiments,dict_features):
  data_x=np.zeros([len(sentiments),len(dict_features)+1])
  data_y=np.zeros(len(sentiments))
  for i,line in enumerate(sentiments):
    data_x[i]=extract_features(line[3:],dict_features)
    if line[0:2]=="+1":
      data_y[i]=1
  return data_x,data_y

def learn(x,t,eta,epoch):
  w=np.random.random(size=(x.shape[1]))
  c=cost(x,w,t)
  print("\t学習開始\tcost:{}".format(c))
  for i in range(1,epoch+1):
    grad=gradient(x,w,t)
    w-=eta*grad
    if i%100==0:
      c = cost(x, w, t)
      e = np.max(np.absolute(eta * grad))
      print('\t学習中(#{})\tcost：{}\tE:{}'.format(i, c, e))
  c = cost(x, w, t)
  e = np.max(np.absolute(eta * grad))
  print('\t学習完了(#{}) \tcost：{}\tE:{}'.format(i, c, e))
  return w

dict_features = load_dict_features()

# 学習対象の行列と極性ラベルの行列作成
# with codecs.open(fname_sentiment, 'r', fencoding) as file_in:
#     x, y = create_training_set(list(file_in), dict_features)

# # 学習
# print('学習率：{}\t学習繰り返し数：{}'.format(eta, epoch))
# w = learn(x, y, eta=eta, epoch=epoch)
# # 結果を保存
# np.save(fname_theta, w)

#74
# theta=np.load(fname_theta)
# review=input('レビューを入力して下さい')
# data_one_x=extract_features(review,dict_features)
# y=sigmoid(data_one_x,theta)
# if y > 0.5:
#     print('label:+1 ({})'.format(y))
# else:
#     print('label:-1 ({})'.format(1 - y))

#75
# with codecs.open(fname_features, 'r', fencoding) as file_in:
#     features = list(file_in)

# # 学習結果の読み込み
# theta = np.load(fname_theta)

# # 重みでソートしてインデックス配列作成
# index_sorted = np.argsort(theta)

# print('top 10')
# for index in index_sorted[:-11:-1]:
#     print('\t{}\t{}'.format(theta[index],
#             features[index - 1].strip() if index > 0 else '(none)'))

# print('worst 10')
# for index in index_sorted[0:10:]:
#     print('\t{}\t{}'.format(theta[index],
#             features[index - 1].strip() if index > 0 else '(none)'))

#76
theta = np.load(fname_theta)
with codecs.open(fname_sentiment, 'r', fencoding) as file_in, \
        open(fname_result, 'w') as file_out:
  for line in file_in:
      # 素性抽出
      data_one_x = extract_features(line[3:], dict_features)

      # 予測、結果出力
      y = sigmoid(data_one_x, theta)
      if y > 0.5:
          file_out.write('{}\t{}\t{}\n'.format(line[0:2], '+1', y))
      else:
          file_out.write('{}\t{}\t{}\n'.format(line[0:2], '-1', 1 - y))
#77


def score(fname):
    '''結果ファイルからスコア算出
    結果ファイルを読み込んで、正解率、適合率、再現率、F1スコアを返す

    戻り値：
    正解率,適合率,再現率,F1スコア
    '''
    # 結果を読み込んで集計
    TP = 0      # True-Positive     予想が+1、正解も+1
    FP = 0      # False-Positive    予想が+1、正解は-1
    FN = 0      # False-Negative    予想が-1、正解は+1
    TN = 0      # True-Negative     予想が-1、正解も-1

    with open(fname) as data_file:
        for line in data_file:
            cols = line.split('\t')
            if len(cols) < 3:
                continue
            if cols[0] == '+1':         # 正解はポジティブ
                if cols[1] == '+1':     # 予想
                    TP += 1
                else:
                    FN += 1
            else:                      #正解はネガティブ
                if cols[1] == '+1': FP += 1
                else:
                    TN += 1
    # 算出
    accuracy = (TP + TN) / (TP + FP + FN + TN)      # 正解率
    precision = TP / (TP + FP)      # 適合率（確実にTrueを選びたい）
    recall = TP / (TP + FN)     # 再現率（Trueを逃すのを避けたい。どれだけxが低くても拾いたい）
    f1 = (2 * recall * precision) / (recall + precision)    # F1スコア

    return accuracy, precision, recall, f1


# スコア算出
accuracy, precision, recall, f1 = score(fname_result)
print('正解率　\t{}\n適合率　\t{}\n再現率　\t{}\nF1スコア　\t{}'.format(
    accuracy, precision, recall, f1
))

#78
# division=5
# with codecs.open(fname_sentiment, 'r', fencoding) as file_in:
#     sentiments_all = list(file_in)
# sentiments=[]
# unit=int(len(sentiments_all)/division)
# for i in range(5):
#   sentiments.append(sentiments_all[i*unit:(i+1)*unit])

# with open(fname_result,"w") as file_out:
#   for i in range(division):
#     print('{}/{}'.format(i+1,division))
#     data_learn=[]
#     for j in range(division):
#       if i==j:
#         data_validation=sentiments[j]
#       else:
#         data_learn+=sentiments[j]
#     data_x,data_y=create_training_set(data_learn,dict_features)
#     theta=learn(data_x,data_y,eta=eta,epoch=epoch)

#     for line in data_validation:
#       data_one_x=extract_features(line[3:],dict_features)
#       y=sigmoid(data_one_x,theta)
#       if y > 0.5:
#         file_out.write('{}\t{}\t{}\n'.format(line[0:2], '+1', y))
#       else:
#         file_out.write('{}\t{}\t{}\n'.format(line[0:2], '-1', 1 - y))

# print('\n学習レート：{}\t学習繰り返し数：{}'.format(eta, epoch))
# accuracy, precision, recall, f1 = score(fname_result)
# print('正解率　\t{}\n適合率　\t{}\n再現率　\t{}\nF1スコア　\t{}'.format(
#     accuracy, precision, recall, f1
# ))


#79
results = []
with open(fname_result) as data_file:
    for line in data_file:

        cols = line.split('\t')
        if len(cols) < 3:
            continue

        # 正解ラベル
        label = cols[0]

        # 識別関数predict()の値
        if cols[1] == '-1':
            predict = 1.0 - float(cols[2])      # 確率を戻す
        else:
            predict = float(cols[2])

        results.append((label, predict))

# 閾値を変えながらスコア算出、グラフ描画用の配列へセット
thresholds = []
accuracys = []
precisions = []
recalls = []
f1s = []
for threshold in np.arange(0.02, 1.0, 0.02):

    # score()を使うため、一時ファイルに結果保存
    with open(fname_work, 'w') as file_out:
        for label, predict in results:
            if predict > threshold:
                file_out.write('{}\t{}\t{}\n'.format(label, '+1', predict))
            else:
                file_out.write('{}\t{}\t{}\n'.format(label, '-1', 1 - predict))

    # スコア算出
    accuracy, precision, recall, f1 = score(fname_work)

    # 結果追加
    thresholds.append(threshold)
    accuracys.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)


# グラフで使うフォント情報(デフォルトのままでは日本語が表示できない)
# fp = FontProperties(
#     fname='/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf'
# )

# 折線グラフの値の設定
plt.plot(thresholds, accuracys, color='green', linestyle='--', label='正解率')
plt.plot(thresholds, precisions, color='red', linewidth=3, label='適合率')
plt.plot(thresholds, recalls, color='blue', linewidth=3, label='再現率')
plt.plot(thresholds, f1s, color='magenta', linestyle='--', label='F1スコア')

# 軸の値の範囲の調整
plt.xlim(
    xmin=0, xmax=1.0
)
plt.ylim(
    ymin=0, ymax=1.0
)

# グラフのタイトル、ラベル指定

# グリッドを表示
plt.grid(axis='both')

# 凡例表示
plt.legend(loc='lower left')

# 表示
plt.show()













