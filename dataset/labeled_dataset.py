import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

idx_to_season = {
    0: "autunno",
    1: "inverno",
    2: "primavera",
    3: "estate"
}

idx_to_subtype = {
    0: "deep",
    1: "soft",
    2: "warm",
    3: "cool",
    4: "light",
    5: "bright"
}


class ArmocromiaDataset(Dataset):
    """
    Dataset for structured images

    Returns:
        - image (tensor)
        - season (int)
        - subtype (int)
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.season_to_idx = {v: k for k, v in idx_to_season.items()}

        self.subtype_to_idx = {v: k for k, v in idx_to_subtype.items()}

        self.images_labeled= []

        for season in os.listdir(root_dir):
            season_path = os.path.join(root_dir, season)
            if not os.path.isdir(season_path):
                continue

            
            if season not in self.season_to_idx:
                continue

            for subtype in os.listdir(season_path):
                subtype_path = os.path.join(season_path, subtype)
                if not os.path.isdir(subtype_path):
                    continue

                
                if subtype not in self.subtype_to_idx:
                    continue

                for fname in os.listdir(subtype_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(subtype_path, fname)

                        season_idx = self.season_to_idx[season]
                        subtype_idx = self.subtype_to_idx[subtype]

                        self.images_labeled.append((img_path, season_idx, subtype_idx))


    def __len__(self):
        return len(self.images_labeled)


    def __getitem__(self, idx):
        img_path, season_label, subtype_label = self.images_labeled[idx]

        
        image = Image.open(img_path).convert("RGB")

        
        if self.transform is not None:
            image = self.transform(image)

        return image, season_label, subtype_label



def get_default_transforms(train=True):
    if train:
        return transforms.Compose([
    
            transforms.Resize((128,128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),


            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


