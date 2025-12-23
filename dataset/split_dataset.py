import os
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png")


def leaf_folders(split_dir):
    leaves = {}
    if not os.path.isdir(split_dir):
        return leaves

    for season in sorted(os.listdir(split_dir)):
        season_path = os.path.join(split_dir, season)
        if not os.path.isdir(season_path):
            continue

        for subtype in sorted(os.listdir(season_path)):
            subtype_path = os.path.join(season_path, subtype)
            if os.path.isdir(subtype_path):
                leaves[(season, subtype)] = subtype_path

    return leaves


def list_images(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXT)
    ])


def count_images_in_split(split_dir):
    total = 0
    for _, folder in leaf_folders(split_dir).items():
        total += len(list_images(folder))
    return total


def allocate_quotas(counts_dict, total_to_move):
    keys = sorted(counts_dict.keys())
    total_available = sum(counts_dict[k] for k in keys)

    if total_to_move <= 0 or total_available <= 0:
        return {k: 0 for k in keys}

    quotas = {}
    for k in keys:
        quotas[k] = int((counts_dict[k] / total_available) * total_to_move)

    assigned = sum(quotas.values())
    remainder = total_to_move - assigned

    if remainder > 0:
        candidates = sorted(
            keys,
            key=lambda k: (-(counts_dict[k] - quotas[k]), k)
        )
        i = 0
        while remainder > 0 and i < len(candidates) * 10:
            k = candidates[i % len(candidates)]
            if quotas[k] < counts_dict[k]:
                quotas[k] += 1
                remainder -= 1
            i += 1

    for k in keys:
        quotas[k] = min(quotas[k], counts_dict[k])

    assigned = sum(quotas.values())
    short = total_to_move - assigned
    if short > 0:
        donors = [k for k in keys if quotas[k] < counts_dict[k]]
        for k in donors:
            take = min(short, counts_dict[k] - quotas[k])
            quotas[k] += take
            short -= take
            if short == 0:
                break

    return quotas


def move_first_k_per_leaf(src_split_dir, dst_split_dir, quotas):
    src_leaves = leaf_folders(src_split_dir)

    for (season, subtype), k in sorted(quotas.items()):
        if k <= 0:
            continue

        src_folder = src_leaves.get((season, subtype))
        if not src_folder or not os.path.isdir(src_folder):
            continue

        imgs = list_images(src_folder)
        to_move = imgs[:k]

        dst_folder = os.path.join(dst_split_dir, season, subtype)
        os.makedirs(dst_folder, exist_ok=True)

        for fname in to_move:
            src = os.path.join(src_folder, fname)
            dst = os.path.join(dst_folder, fname)
            shutil.move(src, dst)


def rebalance_80_10_10(root_dir="data_raw"):
    train_dir = os.path.join(root_dir, "train")
    test_dir  = os.path.join(root_dir, "test")
    val_dir   = os.path.join(root_dir, "val")

    n_train0 = count_images_in_split(train_dir)
    n_test0  = count_images_in_split(test_dir)
    total0   = n_train0 + n_test0

    target = int(0.10 * total0)

    print(f"[START] train={n_train0}  test={n_test0}  total={total0}")
    print(f"[TARGET]  val=test={target}  train={total0 - 2*target}")

    train_leaves = leaf_folders(train_dir)
    train_counts = {k: len(list_images(p)) for k, p in train_leaves.items()}
    quotas_train_to_val = allocate_quotas(train_counts, target)

    move_first_k_per_leaf(train_dir, val_dir, quotas_train_to_val)

    n_test1 = count_images_in_split(test_dir)
    excess = n_test1 - target

    if excess > 0:
        test_leaves = leaf_folders(test_dir)
        test_counts = {k: len(list_images(p)) for k, p in test_leaves.items()}
        quotas_test_to_train = allocate_quotas(test_counts, excess)

        move_first_k_per_leaf(test_dir, train_dir, quotas_test_to_train)

    n_trainF = count_images_in_split(train_dir)
    n_valF   = count_images_in_split(val_dir)
    n_testF  = count_images_in_split(test_dir)
    totalF   = n_trainF + n_valF + n_testF

    print(f"[END]   train={n_trainF}  val={n_valF}  test={n_testF}  total={totalF}")

    if n_valF != target or n_testF != target:
        print("WARNING: target not reached.")
    if n_valF != n_testF:
        print("WARNING: val and test has not the same length.")


if __name__ == "__main__":
    rebalance_80_10_10(root_dir="data_raw")