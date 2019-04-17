#60
import re
import snowballstemmer
import subprocess
import xml.etree.ElementTree as ET
import os
fname="nlp.txt"
fname_parsed="nlp.txt.xml"

def nlp_lines():
  with open('nlp.txt') as lines:
    #[||]指定した語のどれか
    #()かっこに入っている語をまとめて語としてくれる
    pattern=re.compile(r'''(^.*?[\.|;|:|\?|!])\s([A-Z].*)''',re.MULTILINE+re.VERBOSE+re.DOTALL)
    for line in lines:
      line=line.strip()
      while len(line)>0:
        match=pattern.match(line)
        if match:
          yield match.group(1)
          line=match.group(2)
        else:
          yield line
          line=""
# for line in nlp_lines():
#   print(line)
#61
# def nlp_words():
#   tmp=list(map(lambda x:x.split(),[line for line in nlp_lines()]))
#   vocabs=set()
#   for vs in tmp:
#     for vocab in vs:
#       vocabs.add(vocab)
#   vocabs=list(map(lambda x:x.strip('.,;:?!)('),list(vocabs)))
#   for word in vocabs:
#     yield word
#62
stemmer=snowballstemmer.stemmer('english')
# for word in nlp_words():
#   print('{}\t{}'.format(word,stemmer.stemWord(word)))

################63以降でエラーが出て実装ができない###########################33
def parse_nlp():
  if not os.path.exists(fname_parsed):
    subprocess.run(
    'java -cp "/Users/ryousuke/Desktop/nlp100/stanford-corenlp-full-2018-10-05/*"'
    ' -Xmx2g'
    ' edu.stanford.nlp.pipeline.StanfordCoreNLP'
    ' -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref'
    ' -file ' + fname + '>parse.out',
    shell=True,
    check=True
      )
parse_nlp()
root=ET.parse(fname_parsed)
for word in root.iter('word'):
  print(word.text)
