#################################
###           seek        ######
###################################



#書き込んだりした後は、場所が文字の一番最後になっている。そのため例えば、最初から読み込みたいという時には
#一番最初に戻してあげる必要がある。その元に戻す方法が「f.seek("戻したい場所")」と
#記述することができる。