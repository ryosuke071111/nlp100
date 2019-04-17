#20
import json
import re
def uk():
  f=open('jawiki-country.json')
  f=f.readlines()
  for i in f:
    i=json.loads(i)
    if i['title']=='イギリス':
      return(i['text'])
print(uk())
#21
pattern = re.compile(r'''^\[\[Category:.*\]\]''', re.MULTILINE + re.VERBOSE)
result=pattern.findall(uk())
for line in result:
  print(line)
#22
pattern = re.compile(r'''
    \[\[Category:(.*?)(?:\|.*)?\]\]''', re.MULTILINE + re.VERBOSE)
result=pattern.findall(uk())
for line in result:
  print(line)
#23

#24
pattern=re.compile(r'''(?:File|ファイル):(.+?)\|''')
result=pattern.findall(uk())
for line in result:
  print(line)
#25
pattern = re.compile(r'''
    ^\{\{基礎情報.*?$   # '{{基礎情報'で始まる行
    (.*?)       # キャプチャ対象、任意の0文字以上、非貪欲
    ^\}\}$      # '}}'の行
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)
contents = pattern.findall(uk())
pattern = re.compile(r'''
    ^\|         # '|'で始まる行
    (.+?)       # キャプチャ対象（フィールド名）、任意の1文字以上、非貪欲
    \s*         # 空白文字0文字以上
    =
    \s*         # 空白文字0文字以上
    (.+?)       # キャプチャ対象（値）、任意の1文字以上、非貪欲
    (?:         # キャプチャ対象外のグループ開始
        (?=\n\|)    # 改行+'|'の手前（肯定の先読み）
        | (?=\n$)   # または、改行+終端の手前（肯定の先読み）
    )           # グループ終了
    ''', re.MULTILINE + re.VERBOSE + re.DOTALL)
fields = pattern.findall(contents[0])
result = {}
keys_test = []      # 確認用の出現順フィールド名リスト
for field in fields:
    result[field[0]] = field[1]
    keys_test.append(field[0])
for item in sorted(result.items(),
        key=lambda field: keys_test.index(field[0])):
    print(item)
#26
#27
#28
#29
