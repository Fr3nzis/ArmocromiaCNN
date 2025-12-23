import torch

def build_compatibility_matrix(device):
    """
    Stagioni:
    0 = Autunno
    1 = Estate
    2 = Inverno
    3 = Primavera

    Sottotipi:
    0 = Deep
    1 = Soft
    2 = Warm
    3 = Light
    4 = Cool
    5 = Bright
    """

    M = torch.zeros(4, 6, device=device)

    M[0, [0, 1, 2]] = 1  # Autunno
    M[1, [1, 3, 4]] = 1  # Estate
    M[2, [0, 4, 5]] = 1  # Inverno
    M[3, [2, 3, 5]] = 1  # Primavera

    return M