import torch
import torch.nn as nn
import numpy as np


class CLAP_Loss:
    def __init__(self, temp, device):
        self.criterion = nn.CrossEntropyLoss()
        self.temp = temp
        self.device = device

    def get_loss(self, audio_embeds, text_embeds, mrl_dim):
        audio_embeds = audio_embeds / torch.norm(audio_embeds, dim=1, keepdim=True)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)
        logits = (audio_embeds[:, :mrl_dim] @ text_embeds[:, :mrl_dim].T) * np.exp(
            self.temp
        )
        logits = logits.to(self.device)
        n = logits.shape[0]
        labels = torch.arange(n).to(self.device)
        loss_i = self.criterion(logits, labels)
        loss_t = self.criterion(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        return loss
