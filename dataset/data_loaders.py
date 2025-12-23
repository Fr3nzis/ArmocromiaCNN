import os
from torch.utils.data import DataLoader
from dataset.labeled_dataset import ArmocromiaDataset, get_default_transforms

DATA = 'data_gray'

def get_dataloaders(data_root=DATA, batch_size=32, num_workers=0):
    """
    Creates DataLoader for train, val, test.
    """


    train_transform = get_default_transforms(train=True)   # augmentation
    test_transform  = get_default_transforms(train=False)  # NO augmentation

   
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    train_dataset = ArmocromiaDataset(train_dir, transform=train_transform)
    val_dataset   = ArmocromiaDataset(val_dir,   transform=test_transform)
    test_dataset  = ArmocromiaDataset(test_dir,  transform=test_transform)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
