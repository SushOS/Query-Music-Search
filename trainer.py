import torch
from torch import optim
from tqdm import tqdm

from loss import CLAP_Loss
from model import get_embeds


def train_clap(model, train_loader, val_loader, num_epochs, MRL_DIMS, lr, temperature):
    train_losses = []
    val_losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap_loss = CLAP_Loss(temperature, device)
    optimizer = optim.Adam(model.clap.parameters(), lr)

    min_loss = 1e9
    for epoch in range(num_epochs):
        model.clap.train()

        epoch_loss = 0
        for audio, text in tqdm(train_loader):
            try:
                optimizer.zero_grad()

                audio_embeds, text_embeds = get_embeds(model, audio, text)
                loss = 0
                for dim in MRL_DIMS:
                    loss += clap_loss.get_loss(audio_embeds, text_embeds, dim)
                loss /= len(MRL_DIMS)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            except:
                print("Exception Occurred!")
                continue

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Train Loss at epoch {epoch+1} is {epoch_loss}")

        epoch_val_loss = 0
        for audio, text in tqdm(val_loader):
            try:
                with torch.no_grad():
                    audio_embeds, text_embeds = get_embeds(model, audio, text)
                    loss = 0
                    for dim in MRL_DIMS:
                        loss += clap_loss.get_loss(audio_embeds, text_embeds, dim)
                    loss /= len(MRL_DIMS)
                    epoch_val_loss += loss.item()
            except:
                print("Exception Occurred!")
                continue
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        print(f"Validation Loss at epoch {epoch+1} is {epoch_val_loss}")

        if min_loss > val_losses[-1]:
            min_loss = val_losses[-1]
            torch.save(model.clap.state_dict(), "best_model.pt")

    return model, train_losses, val_losses
