#40
import CaboCha
import re
import pydot_ng as pydot
fname="neko.txt"
fname_parsed="neko.txt.cabocha"
fname_result="result.txt"

def parse_neko():
  with open(fname) as data_file, open(fname_parsed,mode="w") as out_file:
    cabocha=CaboCha.Parser() #parserインスタンス
    for line in data_file:
      out_file.write(cabocha.parse(line).toString(CaboCha.FORMAT_LATTICE)) #構文解析実行

class Morph:
  def __init__(self,surface,base,pos,pos1):
    self.surface=surface
    self.base=base
    self.pos=pos
    self.pos1=pos1

  def __str__(self):
    return "surface[{}]\tbase[{}]\tpos[{}]\tpos1[{}]".format(self.surface,self.base,self.pos,self.pos1)

def neco_lines():
  with open(fname_parsed) as file_parsed:
    morphs=[]
    for line in file_parsed:
      if line=="EOS\n":
        yield morphs
        morphs=[]
      else:
        if line[0]=="*":
          continue
        cols=line.split('\t')
        res_cols=cols[1].split(',')
        morphs.append(Morph(cols[0],res_cols[6],res_cols[0],res_cols[1]))
    raise StopIteration
parse_neko()
for i,morphs in enumerate(neco_lines(),1):
  if i==3:
    for morph in morphs:
      print(morph)
    break
#41

#文節クラス
class Chunk:
  def __init__(self):
    self.morphs=[]
    self.srcs=[]
    self.dst=-1

  def __str__(self):
    surface=""
    for morph in self.morphs:
      surface+=morph.surface
    return '{}\tsrcs{}\tdst[{}]'.format(surface, self.srcs, self.dst)

  def normalized_surface(self):
    result=""
    for morph in self.morphs:
      if morph.pos!="記号":
        result+=morph.surface
    return result

#指定した品詞を含むかチェックする
  def chk_pos(self,pos):
    for morph in self.morphs:
      if morph.pos==pos:
        return True
    return False

  def get_morphs_by_pos(self,pos,pos1=""):
    if len(pos1)>0:
      return [res for res in self.morphs if res.pos== pos and res.pos1==pos1]
    else:
      return [res for res in self.morphs if res.pos==pos]

  def get_kaku_prt(self):
    prts=self.get_morphs_by_pos('助詞')
    if len(prts)>1:
      kaku_prts=self.get_morphs_by_pos('助詞','格助詞')
      if len(kaku_prts)>0:
        prts=kaku_prts
    if len(prts)>0:
      return prts[-1].surface
    else:
      return ""

  def get_sahen(self):
    for i,morph in enumerate(self.morphs[0:-1]):
      if morph.pos=="名詞" and morph.pos1=="サ変接続" and self.morphs[i+1].pos=="助詞" and self.morphs[i+1].surface=="を":
        return morph.surface + self.morphs[i+1].surface
    return ""

  def noun_masked_surface(self,mask,dst=False):
    result=""
    for morph in self.morphs:
      if morph.pos!="記号":
        if morph.pos=="名詞":
          result+=mask
          if dst:
            return result
          mask=""
        else:
          result+=morph.surface
    return result

def graph_from_edges_ex(edge_list,directed=False):
  if directed:
    graph=pydot.Dot(graph_type="digraph")
  else:
    graph=pydot.Dot(graph_type="graph")
  for edge in edge_list:
    id1=str(edge[0][0])
    label1=str(edge[0][1])
    id2=str(edge[1][0])
    label2=str(edge[1][1])
    graph.add_node(pydot.Node(id1,label=label1))
    graph.add_node(pydot.Node(id2,label=label2))
    graph.add_edge(pydot.Edge(id1,id2))
  return graph


def neco_lines():
  with open(fname_parsed) as file_parsed:
    chunks=dict()
    idx=-1
    for line in file_parsed:
      if line=="EOS\n":
        if len(chunks)>0:
          sorted_tuple=sorted(chunks.items(),key=lambda x:x[0])
          yield list(zip(*sorted_tuple))[1]
          chunks.clear()
        else:
          yield[]
      elif line[0]=="*":
        cols=line.split()
        idx=int(cols[1])
        dst=int(re.search(r"(.*?)D",cols[2]).group(1))

        if idx not in chunks:
          chunks[idx]=Chunk()
        chunks[idx].dst=dst

        if dst!=-1:
          if dst not in chunks:
            chunks[dst]=Chunk()
          chunks[dst].srcs.append(idx)
      else:
        cols=line.split('\t')
        res_cols=cols[1].split(',')
        chunks[idx].morphs.append(Morph(cols[0],res_cols[6],res_cols[0],res_cols[1]))
    raise StopIteration
parse_neko()
# for i,chunks in enumerate(neco_lines(),1):
#   if i==8:
#     for j,chunk in enumerate(chunks):
#       print('[{}]{}'.format(j,chunk))
#     break

# #43
# for chunks in neco_lines():
#   for chunk in chunks:
#     if chunk.dst!=-1:
#       src=chunk.normalized_surface()
#       dst=chunks[chunk.dst].normalized_surface()
#       if src!="" and dst!="":
#         print('{}\t{}'.format(src, dst))

# #44
# with open(fname,mode="w") as out_file:
#   out_file.write(input('文字列を入力してください-->'))
# parse_neko()
# for chunks in neco_lines():
#   edges = []
#   for i, chunk in enumerate(chunks):
#     if chunk.dst != -1:
#       src = chunk.normalized_surface()
#       dst = chunks[chunk.dst].normalized_surface()
#       if src != '' and dst != '':
#         edges.append(((i, src), (chunk.dst, dst)))
#   if len(edges) > 0:
#     graph = graph_from_edges_ex(edges, directed=True)
#     graph.write_png('result.png')

# #45
# parse_neko()
# with open(fname_result,mode="w") as out_file:
#   for chunks in neco_lines():
#     for chunk in chunks:
#       verbs=chunk.get_morphs_by_pos('動詞')
#       if len(verbs)<1:
#         continue
#       prts=[]
#       for src in chunk.srcs:
#         prts_in_chunk=chunks[src].get_morphs_by_pos('助詞')
#         if len(prts_in_chunk)>1:
#           kaku_prts=chunks[src].get_morphs_by_pos('助詞','格助詞')
#           if len(kaku_prts)>0:
#             prts_in_chunk=kaku_prts
#         if len(prts_in_chunk)>0:
#           prts.append(prts_in_chunk[-1])
#       if len(prts)<1:
#         continue
#       out_file.write('{}\t{}\n'.format(verbs[0].base,      # 最左の動詞の基本系
#           ' '.join(sorted(prt.surface for prt in prts))   # 助詞は辞書順
#       ))

# #46
# parse_neko()
# with open(fname_result,mode="w") as out_file:
#   for chunks in neco_lines():
#     for chunk in chunks:
#       verbs=chunk.get_morphs_by_pos('動詞')
#       if len(verbs)<1:
#         continue
#       chunks_include_prt=[]
#       for src in chunk.srcs:
#         if len(chunks[src].get_kaku_prt())>0:
#           chunks_include_prt.append(chunks[src])
#       if len(chunks_include_prt)<1:
#         continue
#       chunks_include_prt.sort(key=lambda x:x.get_kaku_prt())
#       out_file.write('{}\t{}\t{}\n'.format(
#                 verbs[0].base,      # 最左の動詞の基本系
#                 ' '.join([chunk.get_kaku_prt() \
#                         for chunk in chunks_include_prt]),      # 助詞
#                 ' '.join([chunk.normalized_surface() \
#                         for chunk in chunks_include_prt])       # 項
#             ))
#47
# parse_neko()
# with open(fname_result,mode="w") as out_file:
#   for chunks in neco_lines():
#     for chunk in chunks:
#       verbs=chunk.get_morphs_by_pos('動詞')
#       if len(verbs)<1:
#         continue
#       chunks_include_prt=[]
#       for src in chunk.srcs:
#         if len(chunks[src].get_kaku_prt())>0:
#           chunks_include_prt.append(chunks[src])
#       if len(chunks_include_prt)<1:
#         continue
#       sahen=""
#       for chunk_src in chunks_include_prt:
#         sahen=chunk_src.get_sahen()
#         if len(sahen)>0:
#           chunk_remove=chunk_src
#           break
#       if len(sahen)<1:
#         continue
#       chunks_include_prt.remove(chunk_remove)
#       chunks_include_prt.sort(key=lambda x:x.get_kaku_prt())
#       out_file.write('{}\t{}\t{}\n'.format(
#           sahen + verbs[0].base,   # サ変接続名詞+を+最左の動詞の基本系
#           ' '.join([chunk.get_kaku_prt() \
#                   for chunk in chunks_include_prt]),      # 助詞
#           ' '.join([chunk.normalized_surface() \
#                   for chunk in chunks_include_prt])       # 項
#       ))
#48
# parse_neko()
# with open(fname_result,mode="w") as out_file:
#   for chunks in neco_lines():
#     for chunk in chunks:
#       if(len(chunk.get_morphs_by_pos('名詞'))>0):
#         out_file.write(chunk.normalized_surface())
#         dst=chunk.dst
#         while dst!=-1:
#           out_file.write('->'+chunks[dst].normalized_surface())
#           dst=chunks[dst].dst
#         out_file.write('\n')
#49
parse_neko()
with open(fname_result,mode="w") as out_file:
  for chunks in neco_lines():
    indexs_noun=[i for i in range(len(chunks)) if len(chunks[i].get_morphs_by_pos('名詞'))>0]
    if len(indexs_noun)<2:
      continue

    for i,index_x in enumerate(indexs_noun[:-1]):
      for index_y in indexs_noun[i+1:]:
        meet_y=False
        index_dup=-1
        routes_x=set()

        dst=chunks[index_x].dst
        while dst!=-1:
          if dst==index_y:
            meet_y=True
            break
          routes_x.add(dst)
          dst=chunks[dst].dst

        if not meet_y:
          dst=chunks[index_y].dst
          while dst!= -1:
            if dst in routes_x:
              index_dup=dst
              break
            else:
              dst=chunks[dst].dst
        if index_dup==-1:
          out_file.write(chunks[index_x].noun_masked_surface('X'))
          dst=chunks[index_x].dst
          while dst!=-1:
            if dst==index_y:
              out_file.write('->'+chunks[dst].noun_masked_surface('Y',True))
              break
            else:
              out_file.write('->'+chunks[dst].normalized_surface())
            dst=chunks[dst].dst
          out_file.write('\n')
        else:
          out_file.write(chunks[index_x].noun_masked_surface('X'))
          dst = chunks[index_x].dst
          while dst != index_dup:
              out_file.write(' -> ' + chunks[dst].normalized_surface())
              dst = chunks[dst].dst
          out_file.write(' | ')

          # Yからぶつかる手前までを出力
          out_file.write(chunks[index_y].noun_masked_surface('Y'))
          dst = chunks[index_y].dst
          while dst != index_dup:
              out_file.write(' -> ' + chunks[dst].normalized_surface())
              dst = chunks[dst].dst
          out_file.write(' | ')

          # ぶつかったchunkを出力
          out_file.write(chunks[index_dup].normalized_surface())
          out_file.write('\n')

