from PIL import Image
import matplotlib.pyplot as plt

#画像の読み込み
im = Image.open("no_0.jpg")

plt.imshow(im)
plt.show()