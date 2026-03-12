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
import warnings
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=TqdmWarning)
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
#---
waveform, sr = librosa.load(example_training_file, sr=16000)
#---
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
#---
mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)
#---

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
        
        audio = torch.tensor(audio, dtype=torch.float32)
        category = torch.tensor(category, dtype=torch.long)

        return audio.to(device), category.to(device), label
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
#---
train_dataset = WavDataset(train_df, "train")
val_dataset   = WavDataset(val_df, "val")
#---
#
#
#
#
#
#---
audio, category, label = train_dataset[0]
#---

print(f"Shape of audio (features): {audio.shape}")
print(f"Category integer: {category}")
print(f"Category label: {label}")

print(f"Total number of training examples: {len(train_dataset)}")
#
#
#
#
#
#---
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
#---
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
#---
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        
        self.rnn = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = torch.nn.Linear(hidden_size//2, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)

        return out
#---
#
#
#
#
#
model = RNN(input_size=80000, hidden_size=16, num_classes=len(CATEGORIES)).to(device)
#
#
#
#
#
num_params = sum(param.numel() for param in model.parameters())
print(f"Number of parameters in the model: {num_params}")
#
#
#
#
#
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for X_batch, y_batch, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
#
#
#
#
#
#| code-fold: true
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
    ax.set_xticks(ticks=range(len(CATEGORIES)), labels=CATEGORIES, rotation=45)
    ax.set_yticks(ticks=range(len(CATEGORIES)), labels=CATEGORIES)
    plt.colorbar(im, label="Count")
#
#
#
#
#
#| code-fold: true
# training data
acc, loss, cm = evaluate(model, train_loader)
fig, axarr = plt.subplots(1, 2, figsize=(7, 3))
ax = axarr[0]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Training Confusion Matrix\n(accuracy={acc:.2f})")

acc, loss, cm = evaluate(model, val_loader)
ax = axarr[1]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Validation Confusion Matrix\n(accuracy={acc:.2f})")
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
#
#---
def spectrogram_transform(y): 
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)
#---
#
#
#
#
#
#---
spectrogram_train_dataset = WavDataset(train_df, "train", transform=spectrogram_transform)
spectrogram_val_dataset   = WavDataset(val_df, "val", transform=spectrogram_transform)

spectrogram_train_loader  = DataLoader(spectrogram_train_dataset, batch_size=8, shuffle=True)
spectrogram_val_loader    = DataLoader(spectrogram_val_dataset, batch_size=8, shuffle=False)
#---
#
#
#
#
#
#---
X, y, labels = next(iter(spectrogram_train_loader))
print(X.shape)
#---
#
#
#
#
#
#
#
#| code-fold: true
fig, axarr = plt.subplots(2, 4, figsize=(8, 3), sharex=True, sharey=True)

for i in range(8): 
    axarr.ravel()[i].imshow(X[i].squeeze(), aspect="auto", origin="lower", zorder = 10, cmap = "inferno")
    axarr.ravel()[i].set_title(f"{labels[i]} ({y[i].item()})")
    axarr.ravel()[i].set_xlabel("Time (samples)")
axarr[0, 0].set_ylabel("Mel Frequency Bin")
axarr[1, 0].set_ylabel("Mel Frequency Bin")
plt.tight_layout()
plt.show()
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
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, num_classes)
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
#
#
num_params = sum(param.numel() for param in model.parameters())
print(f"Number of parameters in the model: {num_params}")
#
#
#
#
#
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for X_batch, y_batch, _ in spectrogram_train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
#
#
#
#
#
#
#
acc, loss, cm = evaluate(model, spectrogram_train_loader)
fig, axarr = plt.subplots(1, 2, figsize=(7, 3))
ax = axarr[0]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Training Confusion Matrix\n(accuracy={acc:.2f})")

acc, loss, cm = evaluate(model, spectrogram_val_loader)
ax = axarr[1]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Validation Confusion Matrix\n(accuracy={acc:.2f})")
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
#| code-fold: true
path = "wav_data/train/5-9032-A-0.wav"
audio, sr = librosa.load(path, sr=16000)

spec = spectrogram_transform(audio)

label = metadata[metadata["filename"] == "5-9032-A-0.wav"]["category"].values[0]

fig, ax = plt.subplots(figsize=(7, 3))
im = librosa.display.specshow(spec, y_axis='mel', fmax=8000, x_axis='time', ax = ax)
ax.set_title(f"Mel Spectrogram of {path} ({label})")
t = plt.colorbar(im, label="Magnitude", format='%+2.0f dB')
#
#
#
#
#
#
#| code-fold: true
# translate the spectrogram by 10 time steps
translated_spec = np.roll(spec, shift=50, axis=1)
fig, ax = plt.subplots(figsize=(7, 3))
im = librosa.display.specshow(translated_spec, y_axis='mel', fmax=8000, x_axis='time', ax = ax)
ax.set_title(f"Translated Mel Spectrogram of {path} ({label})")
t = plt.colorbar(im, label="Magnitude", format='%+2.0f dB')
#
#
#
#
#
def transform_pipeline(y):
    
    # previous transform to obtain spectrogram
    spec = spectrogram_transform(y)
    
    # randomly translate the spectrogram by up to 20% of its width in either direction and pad with -80 dB (the minimum value in the spectrogram) on the side that gets rolled over
    max_translation_frac = 0.2
    translation_frac = np.random.uniform(0, max_translation_frac)
    pixels_to_translate = int(translation_frac * spec.shape[1])
    pixels_to_translate = np.random.choice([-pixels_to_translate, pixels_to_translate])
    translated_spec = np.roll(spec, shift=pixels_to_translate, axis=1)
    if pixels_to_translate > 0:
        translated_spec[:, :pixels_to_translate] = -80
    else:
        translated_spec[:, pixels_to_translate:] = -80
    spec_tensor = torch.tensor(translated_spec, dtype=torch.float32)
    
    return spec_tensor
#
#
#
#
#
spectrogram_train_dataset_aug = WavDataset(train_df, "train", transform=transform_pipeline)
spectrogram_val_dataset_aug = WavDataset(val_df, "val", transform=transform_pipeline)

spectrogram_train_loader_aug = DataLoader(spectrogram_train_dataset_aug, batch_size=8, shuffle=True)
spectrogram_val_loader_aug = DataLoader(spectrogram_val_dataset_aug, batch_size=8, shuffle=False)
#
#
#
#
#
first_query = spectrogram_train_dataset_aug[4][0]
second_query = spectrogram_train_dataset_aug[4][0]

print(torch.all(first_query == second_query))
#
#
#
#
#
#
#
#| code-fold: true
fig, axarr = plt.subplots(2, 2, figsize=(7, 3), sharex=True, sharey=True)

for i, ax in enumerate(axarr.flat):
    spec, label, category = spectrogram_train_dataset_aug[4]
    im = librosa.display.specshow(spec.numpy(), y_axis='mel', fmax=8000, x_axis='time', ax = ax, cmap = "inferno")

fig.suptitle(f"Randomly Translated Spectrograms ({category})", fontsize=16)

plt.show()
#
#
#
#
#
#
#
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for X_batch, y_batch, _ in spectrogram_train_loader_aug:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
#
#
#
#
#
acc, loss, cm = evaluate(model, spectrogram_train_loader_aug)
fig, axarr = plt.subplots(1, 2, figsize=(7, 3))
ax = axarr[0]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Training Confusion Matrix\n(accuracy={acc:.2f})")

acc, loss, cm = evaluate(model, spectrogram_val_loader)
ax = axarr[1]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Validation Confusion Matrix\n(accuracy={acc:.2f})")
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
from torchvision import models
model = models.resnet18(weights='IMAGENET1K_V1')
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = torch.nn.Linear(512, len(CATEGORIES)).to(device)
summary(model, (3, 128, 157))
#
#
#
#
#
def transform_pipeline_transfer(y):    
    Y = transform_pipeline(y)
    return torch.tile(Y, dims = (3, 1, 1))
#
#
#
#
#
#| code-fold: true
transfer_dataset_train = WavDataset(train_df, "train", transform=transform_pipeline_transfer)
transfer_dataset_val = WavDataset(val_df, "val", transform=transform_pipeline_transfer)

transfer_loader_train = DataLoader(transfer_dataset_train, batch_size=8, shuffle=True)
transfer_loader_val = DataLoader(transfer_dataset_val, batch_size=8, shuffle=False)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for X_batch, y_batch, _ in transfer_loader_train:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
acc, loss, cm = evaluate(model, transfer_loader_train)
fig, axarr = plt.subplots(1, 2, figsize=(7, 3))
ax = axarr[0]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Training Confusion Matrix\n(accuracy={acc:.2f})")

acc, loss, cm = evaluate(model, transfer_loader_val)
ax = axarr[1]
plot_confusion_matrix(cm, CATEGORIES, ax)
ax.set_title(f"Validation Confusion Matrix\n(accuracy={acc:.2f})")
plt.tight_layout()
#
#
#
#
#
#
#
