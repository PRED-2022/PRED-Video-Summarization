import os
import numpy as np
import pandas as pd


def load_data(dataset, path, split_ratio):
    if split_ratio == 0:
        print("Loading pre-split dataset...")
        return _load_presplit_data(dataset, path)
    else:
        print("Loading dataset...")
        return _load_data(dataset, path, split_ratio)


def load_data_ioc(path, split_ratio):
    """
    Returns the training and validation sets, for images and IOC scores.
    Directory should look like this :
    [base_dir]/datasets/[name_of_dataset]/stimuli/
    [base_dir]/datasets/[name_of_dataset]/ioc.csv
    """

    csv_data = pd.read_csv(os.path.join(path, "CAT2000", "ioc.csv"))

    img_list = csv_data["name"].tolist()
    img_list = [os.path.join(path, "CAT2000", img_name) for img_name in img_list]

    ioc_list = csv_data["ioc"].tolist()

    indices = np.arange(len(img_list))
    np.random.shuffle(indices)
    train_indices = indices[:int(split_ratio*len(img_list))]
    valid_indices = indices[int(split_ratio*len(img_list)):]

    img_train_list = [img_list[idx] for idx in train_indices]
    ioc_train_list = [ioc_list[idx] for idx in train_indices]

    img_val_list = [img_list[idx] for idx in valid_indices]
    ioc_val_list = [ioc_list[idx] for idx in valid_indices]

    assert len(img_train_list) > 0, "No training images were found; check that your directory looks like " \
                                    "[base_dir]/datasets/[name_of_dataset]/stimuli/"
    assert len(img_val_list) > 0, "No validation images were found; check that your directory looks like " \
                                  "[base_dir]/datasets/[name_of_dataset]/stimuli/"
    assert len(ioc_train_list) > 0, "No training saliency maps were found; check that your directory looks like " \
                                    "[base_dir]/datasets/[name_of_dataset]/saliency/"
    assert len(ioc_val_list) > 0, "No validation saliency maps were found; check that your directory looks like " \
                                  "[base_dir]/datasets/[name_of_dataset]/saliency/"

    return img_train_list, img_val_list, ioc_train_list, ioc_val_list


def _load_data(dataset, path, split_ratio):
    """
    Returns the training and validation sets, for images and saliency maps.
    Directory should look like this :
    [base_dir]/datasets/[name_of_dataset]/stimuli/
    [base_dir]/datasets/[name_of_dataset]/saliency/
    """
    img_path = os.path.join(path, "datasets", dataset, "stimuli")
    sal_path = os.path.join(path, "datasets", dataset, "saliency")

    img_list = [os.path.join(img_path, file) for file in os.listdir(img_path) if
                os.path.isfile(os.path.join(img_path, file))]
    sal_list = [os.path.join(sal_path, file) for file in os.listdir(sal_path) if
                os.path.isfile(os.path.join(sal_path, file))]

    check_consistency(zip(img_list, sal_list), len(img_list))

    indices = np.arange(len(img_list))
    np.random.shuffle(indices)
    train_indices = indices[:int(split_ratio*len(img_list))]
    valid_indices = indices[int(split_ratio*len(img_list)):]
    img_train_list = [img_list[idx] for idx in train_indices]
    sal_train_list = [sal_list[idx] for idx in train_indices]
    img_val_list = [img_list[idx] for idx in valid_indices]
    sal_val_list = [sal_list[idx] for idx in valid_indices]

    assert len(img_train_list) > 0, "No training images were found; check that your directory looks like " \
                                    "[base_dir]/datasets/[name_of_dataset]/stimuli/"
    assert len(img_val_list) > 0, "No validation images were found; check that your directory looks like " \
                                  "[base_dir]/datasets/[name_of_dataset]/stimuli/"
    assert len(sal_train_list) > 0, "No training saliency maps were found; check that your directory looks like " \
                                    "[base_dir]/datasets/[name_of_dataset]/saliency/"
    assert len(sal_val_list) > 0, "No validation saliency maps were found; check that your directory looks like " \
                                  "[base_dir]/datasets/[name_of_dataset]/saliency/"

    return img_train_list, img_val_list, sal_train_list, sal_val_list


def _load_presplit_data(dataset, path):
    """
    Returns the training and validation sets, for images and saliency maps,
    when the train-val split is already set.
    Directory should look like this :
    [base_dir]/datasets/[name_of_dataset]/stimuli/train
    [base_dir]/datasets/[name_of_dataset]/stimuli/val
    [base_dir]/datasets/[name_of_dataset]/saliency/train
    [base_dir]/datasets/[name_of_dataset]/saliency/val
    """

    img_train_path = os.path.join(path, "datasets", dataset, "stimuli", "train")
    img_val_path = os.path.join(path, "datasets", dataset, "stimuli", "val")

    sal_train_path = os.path.join(path, "datasets", dataset, "saliency", "train")
    sal_val_path = os.path.join(path, "datasets", dataset, "saliency", "val")

    img_train_list = [os.path.join(img_train_path, file) for file in os.listdir(img_train_path)
                      if os.path.isfile(os.path.join(img_train_path, file))]
    img_val_list = [os.path.join(img_val_path, file) for file in os.listdir(img_val_path)
                    if os.path.isfile(os.path.join(img_val_path, file))]

    sal_train_list = [os.path.join(sal_train_path, file) for file in os.listdir(sal_train_path)
                      if os.path.isfile(os.path.join(sal_train_path, file))]
    sal_val_list = [os.path.join(sal_val_path, file) for file in os.listdir(sal_val_path)
                    if os.path.isfile(os.path.join(sal_val_path, file))]

    assert len(img_train_list) > 0, "No training images were found; check that your directory looks like " \
                                    "[base_dir]/datasets/[name_of_dataset]/stimuli/train"
    assert len(img_val_list) > 0, "No validation images were found; check that your directory looks like " \
                                  "[base_dir]/datasets/[name_of_dataset]/stimuli/val"
    assert len(sal_train_list) > 0, "No training saliency maps were found; check that your directory looks like " \
                                    "[base_dir]/datasets/[name_of_dataset]/saliency/train"
    assert len(sal_val_list) > 0, "No validation saliency maps were found; check that your directory looks like " \
                                  "[base_dir]/datasets/[name_of_dataset]/saliency/val"

    check_consistency(zip(img_train_list, sal_train_list), len(img_train_list))
    check_consistency(zip(img_val_list, sal_val_list), len(img_val_list))

    return img_train_list, img_val_list, sal_train_list, sal_val_list


def define_paths(current_path, args):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.

    Args:
        current_path (str): The absolute path string of this script.
        args (object): A namespace object with values from command line.

    Returns:
        dict: A dictionary with all path elements.
    """

    if os.path.isfile(args.path):
        data_path = args.path
    else:
        data_path = os.path.join(args.path, "")

    results_path = current_path + "/results/"
    weights_path = current_path + "/weights/"

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"

    if args.phase == "train":
        if args.data not in data_path:
            data_path += args.data + "/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path
    }

    return paths


def check_consistency(zipped_file_lists, n_total_files):
    """A consistency check that makes sure all files could successfully be
       found and stimuli names correspond to the ones of ground truth maps.

    Args:
        zipped_file_lists (tuple, str): A tuple of train and valid path names.
        n_total_files (int): The total number of files expected in the list.
    """

    assert len(list(zipped_file_lists)) == n_total_files, "Files are missing"

    for file_tuple in zipped_file_lists:
        file_names = [os.path.basename(entry) for entry in list(file_tuple)]
        file_names = [os.path.splitext(entry)[0] for entry in file_names]
        file_names = [entry.replace("_fixMap", "") for entry in file_names]
        file_names = [entry.replace("_fixPts", "") for entry in file_names]

        assert len(set(file_names)) == 1, "File name mismatch"
