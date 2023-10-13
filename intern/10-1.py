import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import random


# パラメータ
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001

# カスタムデータセット
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# トランスフォーム
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 訓練データのロード
train_dataset = CustomDataset(root_dir='/content/drive/MyDrive/day4/color_relast_2', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# モデル定義
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 損失の履歴を保存するためのリストを初期化
loss_history = []

# 訓練ループ
for epoch in range(EPOCHS):
    total_loss = 0
    for data in train_loader:
        img = data.cuda()
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {average_loss:.4f}")

# 画像の表示
def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(img[:, :, 0], cmap="gray")
    plt.axis('off')

model.eval()
with torch.no_grad():
    for batch in train_loader:
        inputs = batch.cuda()
        outputs = model(inputs)
        break

fig = plt.figure(figsize=(10, 4))
for i in range(5):
    ax = fig.add_subplot(2, 5, i+1)
    imshow(inputs[i])
    ax = fig.add_subplot(2, 5, i+6)
    imshow(outputs[i])
plt.show()

# 損失の推移をプロット
plt.figure(figsize=(10, 6))
plt.plot(loss_history, marker='o', label='Training loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 画像を読み込み
error_image_path = '/content/drive/MyDrive/error/error_4.jpg'
error_image = Image.open(error_image_path)

# トランスフォームを適用
input_tensor = transform(error_image).unsqueeze(0).cuda()  # バッチの次元を追加して、GPUに移動


# モデルを評価モードにし、画像を渡して出力を取得
model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)


# モデルを評価モードにし、画像を渡して出力を取得
model.eval()

# 学習データからランダムに1枚の画像を選びます
random_idx = random.randint(0, len(train_dataset) - 1)
sample_train_image = train_dataset[random_idx]
sample_train_image = sample_train_image.unsqueeze(0).cuda()  # バッチの次元を追加してGPUに転送


# モデルを通して出力を得る
with torch.no_grad():
    reconstructed_error_image = model(input_tensor)  # error_4.jpg を再構築

# lossを計算
loss_between_reconstructed_and_sample = criterion(reconstructed_error_image, sample_train_image)

# lossを表示
print(f"Loss between the reconstructed image from '{error_image_path}' and a random sample from the training data: {loss_between_reconstructed_and_sample.item():.4f}")

# 画像を表示
plt.figure(figsize=(10, 4))

# 入力画像を表示
ax = plt.subplot(1, 2, 1)
imshow(input_tensor[0])
ax.set_title('Input Image')

# 出力画像を表示
ax = plt.subplot(1, 2, 2)
imshow(output_tensor[0])
ax.set_title('Output Image')

plt.tight_layout()
plt.show()
