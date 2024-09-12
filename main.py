import torch
from torch.utils.data import DataLoader
from data import AudioDataset
from model import init_model
from trainer import train_clap

batch_size = 16

audio_folder_path = "/kaggle/input/musicclaps/Audio"
captions_csv_path = "/kaggle/input/caption-data/musiccaps-public.csv"

data = AudioDataset(audio_folder_path, captions_csv_path)
train_data, val_data = torch.utils.data.random_split(data, [0.9, 0.1])

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size)

model = init_model(training_type="Projection Finetune", weights_path=None)
num_epochs = 10
MRL_DIMS = [64, 128, 256, 512]
lr = 1e-4
temperature = model.clap.logit_scale

model, train_losses, val_losses = train_clap(
    model, train_loader, val_loader, num_epochs, MRL_DIMS, lr, temperature
)
