
target_day = int(input("2022年4月のある日付の曜日を出力します。\n日付を入力してください:"))


if 0<=target_day<=31: 
    day = target_day % 7

else:
    print("日付を入力してください")
        

if day is 0:
    print("木曜日です。")

elif day is 1:
    print("金曜日です。")

elif day is 2:
    print("土曜日です。")

elif day is 3:
    print("日曜日です。")

elif day is 4:
    print("月曜日です。")

elif day is 5:
    print("火曜日です。")

elif day is 6:
    print("水曜日です。")