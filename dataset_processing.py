from dataset.split_dataset import rebalance_80_10_10
from dataset.preprocess import process_dataset

if __name__ == "__main__":

    #rebalance_80_10_10(root_dir="data_raw")

    process_dataset(
        src_root="data_raw/train",
        dst_root="data_gray/train"
    )

    process_dataset(
        src_root="data_raw/val",
        dst_root="data_gray/val"
    )

    process_dataset(
        src_root="data_raw/test",
        dst_root="data_gray/test"
    )

    print("\nFINITO! Il dataset preprocessato si trova in data_clean/")