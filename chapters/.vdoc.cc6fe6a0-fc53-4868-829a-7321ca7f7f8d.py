# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import style
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
import librosa
import pandas as pd
import numpy as np
import requests
from matplotlib import pyplot as plt
import os 
import torch
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
metadata_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/refs/heads/master/meta/esc50.csv"
metadata = pd.read_csv(metadata_url)
metadata.head()
#
#
#
#
#
#
#
CATEGORIES = ["dog", "cow", "chirping_birds", "vacuum_cleaner"]
category_dict = {category: idx for idx, category in enumerate(CATEGORIES)}

metadata_subset = metadata[metadata["category"].isin(CATEGORIES)]
train_df = metadata_subset[metadata_subset["fold"] != 1]
val_df = metadata_subset[metadata_subset["fold"] == 1]
#
#
#
#
#
print(f"Number of training examples: {len(train_df)}")
print(f"Number of validation examples: {len(val_df)}")
#
#
#
#
#
#
base_url = "https://github.com/karolpiczak/ESC-50/raw/refs/heads/master/audio"

def download_wav_data(wav_id, data_dir):
    
    # create a directory for each data set if it doesn't exist yet
    if not os.path.exists(f"wav_data/{data_dir}"):
        os.makedirs(f"wav_data/{data_dir}")
    
    # if the file isn't already downloaded, download it and save it to the appropriate directory
    destination = f"wav_data/{data_dir}/{wav_id}"
    if not os.path.exists(destination):
        url = f"{base_url}/{wav_id}"
        response = requests.get(url)
        with open(destination, "wb") as file:
            file.write(response.content)
#
#
#
#
#
res = train_df["filename"].apply(lambda x: download_wav_data(x, "train"))
res = val_df["filename"].apply(lambda x: download_wav_data(x, "val"))
#
#
#
#
#
num_training_files = len(os.listdir("wav_data/train"))
num_validation_files = len(os.listdir("wav_data/val"))

print(f"Number of training files: {num_training_files}")
print(f"Number of validation files: {num_validation_files}")

example_training_file = f"wav_data/train/{os.listdir('wav_data/train')[0]}"

print(f"Example training filename: {example_training_file}")
#
#
#
#
#
#
#
#
#
#
#
waveform, sr = librosa.load(example_training_file, sr=16000)
print(type(waveform))
print(f"Shape of audio array: {waveform.shape}")
print(f"Sampling rate: {sr}")
#
#
#
#
#
#
#
fig, ax = plt.subplots(figsize=(7, 2))
ax.plot(waveform, color = "black", linewidth = .1)
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Waveform of {example_training_file}")
plt.tight_layout()
#
#
#
#
#
#
#
#
#
#
#
mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)

fig, ax = plt.subplots(figsize = (7, 3))
im = librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time', ax = ax)
ax.set_title(f"Mel Spectrogram of {example_training_file}")
plt.colorbar(im, label="Magnitude", format='%+2.0f dB')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}.")
#
#
#
#
#
class WavDataset(Dataset):
    def __init__(self, metadata, path, transform=None):
        self.metadata = metadata #<1> 
        self.transform = transform #<2>
        self.path = path #<3>

    def __len__(self):
        return len(self.metadata) #<4>

    def __getitem__(self, idx):

        # figure out which filename corresponds to the 
        # specified index in self.metadata
        wav_id = self.metadata.iloc[idx]["filename"] 
        wav_path = f"wav_data/{self.path}/{wav_id}"

        # corresponding target value
        label = self.metadata.iloc[idx]["category"] #<5>
        category = category_dict[label]     #<6>
        # load in the audio
        audio, sr = librosa.load(wav_path, sr=16000)
        if self.transform:
            audio = self.transform(audio)
        return audio.to(device), category.to(device), label
```
#
#
#
#
#
#
#
#
#
#
#
#
train_dataset = WavDataset(train_df, "train")
val_dataset   = WavDataset(val_df, "val")
#
#
#
#
#

#
#
#
#
#
#
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
#
#
#
#
#
#
#
#
#
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        
        self.rnn = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
#
#
#
model = RNN(input_size=80000, hidden_size=64, num_classes=len(CATEGORIES)).to(device)
X, y, _ = next(iter(train_loader))
model(X)
#
#
#
#

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for X_batch, y_batch, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        print(loss.item())
#
#
#
#
def evaluate(model, data_loader): 
    confusion_matrix = torch.zeros(len(CATEGORIES), len(CATEGORIES), dtype=torch.int32)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss = 0
    for X, y, labels in data_loader:
        pred = model(X)
        loss += loss_fn(pred, y)
        y_pred = torch.argmax(pred, dim=1)
        for true_label, pred_label in zip(y, y_pred):
            confusion_matrix[true_label, pred_label] += 1
        acc = torch.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return acc.item(), loss.item(), confusion_matrix

def plot_confusion_matrix(cm, categories, ax):
    
    im = ax.imshow(cm, cmap = "inferno", zorder = 10, origin = "lower")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(ticks=range(len(categories)), labels=categories.keys(), rotation=45)
    ax.set_yticks(ticks=range(len(categories)), labels=categories.keys())
    plt.colorbar(im, label="Count")

#
#
#
#
acc, loss, cm = evaluate(model, train_loader)

fig, ax = plt.subplots(figsize=(6, 5))
plot_confusion_matrix(cm, CATEGORIES, ax)

# plt.imshow(cm, cmap = "inferno", zorder = 10, origin = "lower")
#
#
#
#
acc, loss, cm = evaluate(model, val_loader)
fig, ax = plt.subplots(figsize=(6, 5))
plot_confusion_matrix(cm, CATEGORIES, ax)
#
#
#
#
#
#
transform = lambda x: librosa.feature.melspectrogram(y=x, sr=16000, n_mels=128, fmax=8000)

spectrogram_train_dataset = WavDataset(train_df, "train", transform=transform)
spectrogram_val_dataset   = WavDataset(val_df, "val", transform=transform)

spectrogram_train_loader  = DataLoader(spectrogram_train_dataset, batch_size=8, shuffle=True)
spectrogram_val_loader    = DataLoader(spectrogram_val_dataset, batch_size=8, shuffle=False)
#
#
#
X, y, labels = next(iter(spectrogram_train_loader))
print(X.shape)
#
#
#
fig, axarr = plt.subplots(2, 4, figsize=(8, 3), sharex=True, sharey=True)

for i in range(8): 
    axarr.ravel()[i].imshow(X[i].squeeze(), aspect="auto", origin="lower", zorder = 10, cmap = "inferno", vmin = 0, vmax = 50)
    axarr.ravel()[i].set_title(f"{labels[i]} ({y[i].item()})")
    axarr.ravel()[i].set_xlabel("Time (samples)")
axarr[0, 0].set_ylabel("Mel Frequency Bin")
axarr[1, 0].set_ylabel("Mel Frequency Bin")
plt.tight_layout()
plt.show()
#
#
#
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pipeline = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(19456, num_classes)
        )

    def forward(self, x):
        out = x.unsqueeze(1)
        return self.pipeline(out)
#
#
#
model = ConvNet(num_classes=len(CATEGORIES)).to(device)
#
#
#
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(10):
    model.train()
    for X_batch, y_batch, _ in spectrogram_train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Completed epoch {epoch}")
    
    train_acc, train_loss, cm = evaluate(model, spectrogram_train_loader)
    val_acc, val_loss, cm     = evaluate(model, spectrogram_val_loader)

    train_losses += [train_loss]
    train_accs += [train_acc]
    val_losses += [val_loss]
    val_accs += [val_accs]
#
#
#
#
#
acc, loss, cm = evaluate(model, spectrogram_train_loader)
#
#
#
fig, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].plot(train_losses)
ax[1].plot(train_accs)

plot_confusion_matrix(cm, CATEGORIES, ax[2])
plt.tight_layout()
#
#
#
#
acc, loss, cm = evaluate(model, spectrogram_val_loader)
#
#
#
#
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].plot(val_losses)
ax[1].plot(val_accs)

plot_confusion_matrix(cm, CATEGORIES, ax[2])
plt.tight_layout()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
