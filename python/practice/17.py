########################################
######        リストの操作　　　　#######
#########################################



"""
リストの上書き
"""
#インデックス番号を直接指定することで上書きさせることもできる。
s = ["a", "b", "c", "d", "e"]
print(s)
s[0] = "A"
print(s)

#スライスも使うことができる。
s[2:4] = ["C", "D"]
print(s)



"""
リスト追加
"""
# .append とすると、末尾に追加することができる。
X = [2,4,6,8,10]
print(X)
X.append(12)
print(X)

# .insert("挿入したいインデックス番号","追加したい数字") で、指定した箇所に追加することができる。
X.insert(0,1)
print(X)

# .pop("削除したい箇所のインデックス番号") で指定した箇所を削除することができる。
X.pop(3)
print(X)


"""
リストの結合
"""

#単なる足し算のようにリストを結合させることができる。
a = [1,2,3,4,5]
b = [6,7,8,9,10]
c = a + b
print(c)

#メゾットを使うことで結合させることもできる。
x = [1,2,3,4,5]
y = [6,7,8,9,10]

x.extend(y)
print(x)