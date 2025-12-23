import torch
from src.Network.model import CNNMultiTask
from dataset.data_loaders import get_dataloaders
from dataset.compatibility_matrix import build_compatibility_matrix
from src.test import evaluate_model
from src.train import get_device


def test():

    device = get_device()
    _, _, test_loader = get_dataloaders(batch_size=32)

    model = CNNMultiTask(num_stagioni=4, num_sottotipi=6).to(device)
    model.load_state_dict(torch.load("modello_migliore.pth", map_location=device))
    model.eval()
    M = build_compatibility_matrix(device)
    cm = False
    evaluate_model(model,
    test_loader,
    device,
    M,
    cm)


if __name__ == "__main__":
    test()