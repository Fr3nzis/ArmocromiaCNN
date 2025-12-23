import torch


def incompatibility_loss(out_season, out_subtype, M, eps=1e-8):
    pS = torch.softmax(out_season, dim=1)
    pT = torch.softmax(out_subtype, dim=1)
    p_compat = torch.einsum("bi,ij,bj->b", pS, M, pT)
    loss = -torch.log(p_compat + eps)
    return loss.mean()


def run_one_epoch(
    model,
    loader,
    criterion_season,
    criterion_subtype,
    device,
    M,
    lambda_incompat,
    optimizer=None
):

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct_s = 0
    correct_t = 0
    total = 0

    for images, seasons, subtypes in loader:

        images = images.to(device)
        seasons = seasons.to(device)
        subtypes = subtypes.to(device)

        if is_train:
            optimizer.zero_grad()

        if is_train:
            out_s, out_t = model(images)
        else:
            with torch.no_grad():
                out_s, out_t = model(images)

        loss_s = criterion_season(out_s, seasons)
        loss_t = criterion_subtype(out_t, subtypes)
        loss_i = incompatibility_loss(out_s, out_t, M)

        loss = loss_s + loss_t + lambda_incompat * loss_i

        if is_train:
            loss.backward()
            optimizer.step()

        batch = images.size(0)
        running_loss += loss.item() * batch
        total += batch

        correct_s += (out_s.argmax(1) == seasons).sum().item()
        correct_t += (out_t.argmax(1) == subtypes).sum().item()

    return (
        running_loss / total,
        correct_s / total,
        correct_t / total
    )


def train_model(
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
):
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):

        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc_s, train_acc_t = run_one_epoch(
            model,
            train_loader,
            criterion_season,
            criterion_subtype,
            device,
            M,
            lambda_incompat,
            optimizer=optimizer
        )

        print(
            f"TRAIN → loss {train_loss:.4f} | "
            f"stagione {train_acc_s:.3f} | sottotipo {train_acc_t:.3f}"
        )

        val_loss, val_acc_s, val_acc_t = run_one_epoch(
            model,
            val_loader,
            criterion_season,
            criterion_subtype,
            device,
            M,
            lambda_incompat
        )

        print(
            f"VAL   → loss {val_loss:.4f} | "
            f"stagione {val_acc_s:.3f} | sottotipo {val_acc_t:.3f}"
        )

        scheduler.step(val_loss)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("→ Modello migliore salvato")

    print("\nTraining completato")
    print("Best validation loss:", best_val_loss)

    return best_val_loss


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    return device