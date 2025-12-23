import torch
import torch.nn as nn

from src.Network.model import CNNMultiTask
from dataset.data_loaders import get_dataloaders
from dataset.compatibility_matrix import build_compatibility_matrix
from src.train import train_model, get_device
   

def main():
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    model = CNNMultiTask(num_stagioni=4, num_sottotipi=6).to(device)

    criterion_season = nn.CrossEntropyLoss()
    criterion_subtype = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    M = build_compatibility_matrix(device)
    lambda_incompat = 0.5

    num_epochs = 25

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion_season,
        criterion_subtype,
        device,
        M,
        lambda_incompat,
        num_epochs,
    save_path="modello_migliore.pth"
)


if __name__ == "__main__":
    main()