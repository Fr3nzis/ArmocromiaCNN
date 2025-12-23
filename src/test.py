import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def _save_confusion_matrix(conf_matrix, title, filename, tick_labels=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    n = conf_matrix.shape[0]
    plt.xticks(np.arange(n))
    plt.yticks(np.arange(n))

    if tick_labels is not None:
        plt.xticks(np.arange(n), tick_labels, rotation=45, ha="right")
        plt.yticks(np.arange(n), tick_labels)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, int(conf_matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def evaluate_model(
    model,
    test_loader,
    device,
    M,
    cm=False,
    num_seasons=4,
    num_subtypes=6
):
    model.eval()

    correct_season = 0
    correct_subtype = 0
    total = 0
    incompatible = 0

    all_seasons_true = []
    all_seasons_pred = []
    all_subtypes_true = []
    all_subtypes_pred = []

    if cm:
        conf_season = np.zeros((num_seasons, num_seasons), dtype=int)
        conf_subtype = np.zeros((num_subtypes, num_subtypes), dtype=int)

    with torch.no_grad():
        for images, seasons, subtypes in test_loader:

            images = images.to(device)
            seasons = seasons.to(device)
            subtypes = subtypes.to(device)

            out_s, out_t = model(images)

            pred_s = out_s.argmax(dim=1)
            pred_t = out_t.argmax(dim=1)

            correct_season += (pred_s == seasons).sum().item()
            correct_subtype += (pred_t == subtypes).sum().item()

            all_seasons_true.extend(seasons.cpu().numpy())
            all_seasons_pred.extend(pred_s.cpu().numpy())
            all_subtypes_true.extend(subtypes.cpu().numpy())
            all_subtypes_pred.extend(pred_t.cpu().numpy())

            for s, t in zip(pred_s, pred_t):
                if M[int(s), int(t)] == 0:
                    incompatible += 1

            if cm:
                s_true = seasons.cpu().numpy()
                s_pred = pred_s.cpu().numpy()
                t_true = subtypes.cpu().numpy()
                t_pred = pred_t.cpu().numpy()

                for y_true, y_pred in zip(s_true, s_pred):
                    conf_season[int(y_true), int(y_pred)] += 1

                for y_true, y_pred in zip(t_true, t_pred):
                    conf_subtype[int(y_true), int(y_pred)] += 1

            total += images.size(0)

    acc_season = correct_season / total
    acc_subtype = correct_subtype / total
    incompat_rate = incompatible / total

    f1_season = f1_score(all_seasons_true, all_seasons_pred, average="macro")
    f1_subtype = f1_score(all_subtypes_true, all_subtypes_pred, average="macro")

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy Season:   {acc_season:.4f}")
    print(f"Accuracy Subtype:  {acc_subtype:.4f}")
    print(f"F1 Season:        {f1_season:.4f}")
    print(f"F1 Subtype:       {f1_subtype:.4f}")
    print(f"Incompatibility %:   {100 * incompat_rate:.2f}%")
    print("========================")

    if cm:
        _save_confusion_matrix(
            conf_season,
            title="Confusion Matrix - Season",
            filename="confusion_season.png",
            tick_labels=["Autumn", "Winter", "Spring", "Summer"]
        )

        _save_confusion_matrix(
            conf_subtype,
            title="Confusion Matrix - Subtype",
            filename="confusion_subtype.png",
            tick_labels=["Deep", "Bright", "Light", "Soft", "Warm", "Cool"]
        )

        return acc_season, acc_subtype, incompat_rate, f1_season, f1_subtype, conf_season, conf_subtype

    return acc_season, acc_subtype, incompat_rate, f1_season, f1_subtype