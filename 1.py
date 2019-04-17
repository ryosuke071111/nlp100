import random
#00
print("stressed"[::-1])
#01
print("パタトクカシーー"[::2])
#02
print("".join(i+j for i,j in zip('パトカー','タクシー')))
#03
print([len(i) for i in "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.".split()])
#04
a=[1,5,6,7,8,9,15,16,19]
print({(j[0] if i+1 in a else j[1]):i for i,j in enumerate("Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.".split())})
#05
a="I am an NLPer"
print(list(a))
print([a[i]+a[i+1] for i in range(0,len(a)-1,2)])
#06
a="paraparaparadise"
b="paragraph"
A={a[i]+a[i+1] for i in range(0,len(a)-1,2)}
B={b[i]+b[i+1] for i in range(0,len(b)-1,2)}
print(A)
print(B)
print(A|B)
print(A&B)
print(A-B)
print(True if "se" in (A&B) else False)

#07
def f(x,y,z):
  return(str(x)+"時の"+str(y)+"は"+str(z))
print(f(12,"気温",22.4))
#08
def cipher(str):
  ans=""
  for c in str:
    if 97<=ord(c)<=123:
      ans+=chr(219-ord(c))
    else:
      ans+=c
  return ans
print(cipher("deep learning"))
print(cipher(cipher("deep learning")))
#09
string="I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind .".split()
print(list(map(lambda i:i[0]+"".join(random.sample(i[1:-1],len(i)-2))+i[-1] if len(i)>4 else i,string)))
