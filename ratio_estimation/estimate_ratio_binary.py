"""
Estimate importance weights for a given dataset.

Codebase:
    - Choi et al., 2020: https://github.com/ermongroup/fairgen.git


Config:
    gpu_idx: GPU device number to run. (ex:int(4))
    perc: Size of Balanced Dataset/ Size of Biased dataset
    bias_ratio: Major/Minor Ratio in Biased dataset
    bias_factor: 'Y', 'A', 'Both'
    dataset: 'fmnist', 'mnist'
    target_nce: Target nce in Search Mode
    save_cal: If set True, plot calibration curve in Fixed Mode. 
    datapath: path to data directory (e.g., mnist_28_A_41)


Usage:
    - Train classifier (i.e. estimate importance weight) with given hyperparameters
    - Optionally, you can also get calibration curve
    - Save 
        - Best classifier model dict
        - Train dataset / Y / A / importance weight
        - Hyperparameters, mean/max value of importance weights w.r.t groups
            - If bias on A, ordered as [Minor, Major]
            - If bias on both Y and A, ordered as 
                [Major Y's Major A, Major Y's Minor A, 
                Minor Y's Major A, Minor Y's Minor A]
        - Plot of estimated importance weights
    
Notes:
    - Change data directory information before running this code
    - Class label should be changed if not [3, 1] for MNIST, [7, 1] for FMNIST
"""


import torch
import numpy as np
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import shutil
import argparse
import csv
import matplotlib.pyplot as plt


# ======= Class ================================================
class LoopingDataset(Dataset):
    """
    Dataset class to handle indices going out of bounds when training
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= len(self.dataset):
            index = np.random.choice(len(self.dataset))
        item, attr, label = self.dataset.__getitem__(index)
        return item, attr, label


class BagOfDatasets(Dataset):
    """
    Wrapper class over several dataset classes. from @mhw32
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.n = len(datasets)

    def __len__(self):
        lengths = [len(dataset) for dataset in self.datasets]
        return max(lengths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, ...)
        """
        items = []
        attrs = []
        labels = []
        for dataset in self.datasets:
            item = dataset.__getitem__(index)
            if isinstance(item, tuple):
                data = item[0]
                attr = item[1]  # true female/male label
                label = item[2]  # fake data balanced/unbalanced label
            items.append(data)
            labels.append(label)
            attrs.append(attr)

        return items, attrs, labels


class CLSFR_Model(nn.Module):
    def __init__(self):
        super(CLSFR_Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1
        )
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.fc1 = nn.Linear(9216, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        probas = F.softmax(x, dim=1)

        return x, probas


# ================================================================

# ======= Functions ================================================


def calc_max_group_size_single(bias_factor, digit_list, bias):
    """
    Calculate maximum possible Major/Minor Group size with bias.
    Use with single bias setting.

    Args:
        bias_factor: data indicating major/minor
        digit_list: [Major class label, Minor class label]
        bias: Major/Minor bias ratio

    Returns:
        max major size, max minor size
    """
    major = digit_list[0]
    data_size = bias_factor.shape[0]
    curr_major = (bias_factor == major).sum()
    curr_minor = data_size - curr_major

    major_prop = (0.5 * bias / (bias + 1)) + 0.25
    minor_prop = (0.5 / (bias + 1)) + 0.25
    max_size = int(min(curr_major / major_prop, curr_minor / minor_prop))
    return np.floor([max_size * major_prop, max_size * minor_prop]).astype(int)


def calc_max_group_size_multi(bf_data_1, bf_data_2, digit_list, bias):
    """
    Calculate maximum possible Major/Minor Group size with bias.
    Use with multi bias setting.

    Args:
        bf_data_1: first bias factor data indicating major/minor
        bf_data_2: second bias factor data indicating major/minor
        digit_list: list of [Major class label, Minor class label].
                    Note that list should be ordered.
        bias: Major/Minor bias ratio

    Returns:
        max group size of [bf_1_major_bf_2_major, bf_1_major_bf_2_minor,
                            bf_1_minor_bf_2_major, bf_1_minor_bf_2_minor]
    """
    bf_1_major, bf_1_minor = digit_list[0]
    bf_2_major, bf_2_minor = digit_list[1]

    g1_size = ((bf_data_1 == bf_1_major) & (bf_data_2 == bf_2_major)).sum()
    g2_size = ((bf_data_1 == bf_1_major) & (bf_data_2 == bf_2_minor)).sum()
    g3_size = ((bf_data_1 == bf_1_minor) & (bf_data_2 == bf_2_major)).sum()
    g4_size = ((bf_data_1 == bf_1_minor) & (bf_data_2 == bf_2_minor)).sum()

    g1_prop = (0.5 * (bias**2) / ((bias + 1) ** 2)) + 0.125
    g2_prop = g3_prop = (0.5 * bias / ((bias + 1) ** 2)) + 0.125
    g4_prop = (0.5 / ((bias + 1) ** 2)) + 0.125

    max_size = int(
        min(
            [g1_size / g1_prop, g2_size / g2_prop, g3_size / g3_prop, g4_size / g4_prop]
        )
    )
    return np.floor(
        [max_size * g1_prop, max_size * g2_prop, max_size * g3_prop, max_size * g4_prop]
    ).astype(int)


def build_dataset_single(
    data_path, label_path, digit_list, bias_ratio, split, perc=1.0
):
    """
    Given a dataset, build balanced/unbalanced dataset with bias ratio.
    Use in single bias setting.

    Args:
        digit_list: class label of bias factor.
                    Note that it should be ordered as [Major, Minor]
        split: 'train' or 'valid'
        perc: Size of Balanced Dataset/ Size of Biased dataset

    Returns:
        balanced dataset, unbalanced dataset
    """
    data = torch.load(data_path)
    # (size, 28, 28) -> (size, 1, 28, 28)
    data = data.unsqueeze(1)
    z = torch.load(label_path)
    valid_size = z.shape[0] // 10

    if split == "train":
        data, z = data[valid_size:], z[valid_size:]
    elif split == "valid":
        data, z = data[:valid_size], z[:valid_size]

    major_size, minor_size = calc_max_group_size_single(z, digit_list, bias_ratio)
    total_examples = major_size + minor_size
    data_major_idx = np.where((z == digit_list[0]))[0][:major_size]
    data_minor_idx = np.where((z == digit_list[1]))[0][:minor_size]

    # construct unbiased dataset
    n_balanced = total_examples // 2
    to_stop = int((n_balanced // 2) * perc)
    balanced_indices = np.hstack((data_major_idx[0:to_stop], data_minor_idx[0:to_stop]))
    balanced_dataset = data[balanced_indices]
    balanced_zs = z[balanced_indices]
    # print('balanced dataset ratio: {}'.format(np.unique(balanced_zs.numpy(), return_counts=True)))

    # construct biased dataset
    unbalanced_indices = np.hstack(
        (data_major_idx[(n_balanced // 2) :], data_minor_idx[(n_balanced // 2) :])
    )
    unbalanced_dataset = data[unbalanced_indices]
    unbalanced_zs = z[unbalanced_indices]
    # print('unbalanced dataset ratio: {}'.format(np.unique(unbalanced_zs.numpy(), return_counts=True)))

    # print('converting attribute zs to balanced/unbalanced...')
    data_balanced_labels = torch.ones_like(balanced_zs)  # y = 1 for balanced
    data_unbalanced_labels = torch.zeros_like(unbalanced_zs)  # y = 0 for unbalanced

    # construct dataloaders
    balanced_train_dataset = torch.utils.data.TensorDataset(
        balanced_dataset, balanced_zs, data_balanced_labels
    )
    unbalanced_train_dataset = torch.utils.data.TensorDataset(
        unbalanced_dataset, unbalanced_zs, data_unbalanced_labels
    )

    # make sure things don't go out of bounds during trainin
    balanced_train_dataset = LoopingDataset(balanced_train_dataset)
    unbalanced_train_dataset = LoopingDataset(unbalanced_train_dataset)

    return balanced_train_dataset, unbalanced_train_dataset


def build_dataset_multi(
    data_path, label_path, z_path, digit_list, bias_ratio, split, perc=1.0
):
    """
    Given a dataset, build balanced/unbalanced dataset with bias ratio.
    Use in multi bias setting.

    Groups Label:
        0: bf_1_major_bf_2_major
        1: bf_1_major_bf_2_minor
        2: bf_1_minor_bf_2_major
        3: bf_1_minor_bf_2_minor

    Args:
        digit_list: list of class label of bias factor.
                    Note that it should be ordered as [Major, Minor]
        split: 'train' or 'valid'
        perc: Size of Balanced Dataset/ Size of Biased dataset

    Returns:
        balanced dataset, unbalanced dataset
    """
    data = torch.load(data_path)
    # (size, 28, 28) -> (size, 1, 28, 28)
    data = data.unsqueeze(1)
    label = torch.load(label_path)
    z = torch.load(z_path)
    valid_size = z.shape[0] // 10

    # obtain proper number of examples
    if split == "train":
        data, label, z = data[valid_size:], label[valid_size:], z[valid_size:]
    elif split == "valid":
        data, label, z = data[:valid_size], label[:valid_size], z[:valid_size]

    g1_size, g2_size, g3_size, g4_size = calc_max_group_size_multi(
        label, z, digit_list, bias_ratio
    )
    total_examples = g1_size + g2_size + g3_size + g4_size

    l_major, l_minor = digit_list[0]
    z_major, z_minor = digit_list[1]
    cln_3 = np.where((z == z_major) & (label == l_major))[0][:g1_size]
    rot_3 = np.where((z == z_minor) & (label == l_major))[0][:g2_size]
    cln_1 = np.where((z == z_major) & (label == l_minor))[0][:g3_size]
    rot_1 = np.where((z == z_minor) & (label == l_minor))[0][:g4_size]

    # construct balanced dataset
    n_balanced = total_examples // 2

    to_stop = int((n_balanced // 4) * perc)  # 4 categories
    balanced_indices = np.hstack(
        (cln_1[0:to_stop], cln_3[0:to_stop], rot_1[0:to_stop], rot_3[0:to_stop])
    )
    balanced_dataset = data[balanced_indices]
    balanced_zs = torch.Tensor(to_stop * [2, 0, 3, 1])
    # print('balanced dataset ratio: {}'.format(np.unique(balanced_zs.numpy(), return_counts=True)))

    unbalanced_indices = np.hstack(
        (
            cln_1[(n_balanced // 4) :],
            cln_3[(n_balanced // 4) :],
            rot_1[(n_balanced // 4) :],
            rot_3[(n_balanced // 4) :],
        )
    )
    unbalanced_dataset = data[unbalanced_indices]
    unbalanced_zs = torch.Tensor(
        np.hstack(
            (
                len(cln_1[(n_balanced // 4) :]) * [2],
                len(cln_3[(n_balanced // 4) :]) * [0],
                len(rot_1[(n_balanced // 4) :]) * [3],
                len(rot_3[(n_balanced // 4) :]) * [1],
            )
        )
    )
    # print('unbalanced dataset ratio: {}'.format(np.unique(unbalanced_zs, return_counts=True)))

    # print('converting attribute labels to balanced/unbalanced...')
    data_balanced_labels = torch.ones_like(balanced_zs)  # y = 1 for balanced
    data_unbalanced_labels = torch.zeros_like(unbalanced_zs)  # y = 0 for unbalanced

    # construct dataloaders
    balanced_train_dataset = torch.utils.data.TensorDataset(
        balanced_dataset, balanced_zs, data_balanced_labels
    )
    unbalanced_train_dataset = torch.utils.data.TensorDataset(
        unbalanced_dataset, unbalanced_zs, data_unbalanced_labels
    )

    # make sure things don't go out of bounds during training
    balanced_train_dataset = LoopingDataset(balanced_train_dataset)
    unbalanced_train_dataset = LoopingDataset(unbalanced_train_dataset)

    return balanced_train_dataset, unbalanced_train_dataset


def shrink_valid_dataset(
    balanced_valid_dataset, unbalanced_valid_dataset, bias_factor, bias, digit_list
):
    """
    Shrink Validation set according to the right proportions.
    """
    to_shrink = len(balanced_valid_dataset)
    d, g, l = unbalanced_valid_dataset.dataset.tensors

    if bias_factor == "Both":
        # print('balancing for multi-attribute')
        a = torch.where(g == 0)[0]
        b = torch.where(g == 1)[0]
        c = torch.where(g == 2)[0]
        e = torch.where(g == 3)[0]

        # get indices w.r.t bias ratio
        a_idx = int(to_shrink * ((bias**2) / ((bias + 1) ** 2)))
        b_idx = int(to_shrink * (bias / ((bias + 1) ** 2)))
        c_idx = int(to_shrink * (bias / ((bias + 1) ** 2)))
        e_idx = int(to_shrink * (1 / ((bias + 1) ** 2)))

        # get all indices
        a_idx = a[0:a_idx]
        b_idx = b[0:b_idx]
        c_idx = c[0:c_idx]
        e_idx = e[0:e_idx]

        # aggregate all data
        d = torch.cat([d[a_idx], d[b_idx], d[c_idx], d[e_idx]])
        l = torch.cat([l[a_idx], l[b_idx], l[c_idx], l[e_idx]])
        g = torch.cat([g[a_idx], g[b_idx], g[c_idx], g[e_idx]])

    else:
        major = torch.where(g == digit_list[0])[0]
        males = torch.where(g == digit_list[1])[0]

        f_idx = int(to_shrink * (bias / (bias + 1)))
        m_idx = int(to_shrink * (1 / (bias + 1)))

        f_idx = major[0:f_idx]
        m_idx = males[0:m_idx]

        # aggregate all data
        d = torch.cat([d[f_idx], d[m_idx]])
        l = torch.cat([l[f_idx], l[m_idx]])
        g = torch.cat([g[f_idx], g[m_idx]])

    # print('Changed dataset ratio: {}'.format(np.unique(g.numpy(), return_counts=True)))
    return d, g, l


def train(model, epoch, curr_train_loader, optimizer, device):
    model.train()
    correct = 0.0
    num_examples = 0.0
    preds = []

    for batch_idx, data_list in enumerate(curr_train_loader):
        # concatenate data and labels from both balanced + unbalanced, and make sure that each minibatch is balanced
        n_unbalanced = len(data_list[0][1])
        data = torch.cat((data_list[0][0][0:n_unbalanced], data_list[0][1])).to(device)
        attr = torch.cat((data_list[1][0][0:n_unbalanced], data_list[1][1])).to(device)
        target = torch.cat((data_list[2][0][0:n_unbalanced], data_list[2][1])).to(
            device
        )

        # random permutation of data
        idx = torch.randperm(len(data))
        data = data[idx]
        target = target[idx]
        attr = attr[idx]

        # minor adjustments
        data = data.float() / 255.0
        target = target.long()

        # NOTE: here, balanced (y=1) and unbalanced (y=0)
        logits, probas = model(data)
        loss = F.cross_entropy(logits, target)
        _, pred = torch.max(probas, 1)

        # check performance
        num_examples += target.size(0)
        correct += (pred == target).sum()
        preds.append(pred)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log performance
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(curr_train_loader.dataset),
                    100.0 * batch_idx / len(curr_train_loader),
                    loss.item(),
                )
            )

    # aggregate results
    train_acc = float(correct) / num_examples
    preds = torch.cat(preds)
    preds = np.ravel(preds.data.cpu().numpy())

    return train_acc, preds, loss.item()


def test(model, loader, device):
    model.eval()
    test_loss = 0.0
    num_examples = 0.0
    num_pos_correct = 0.0
    num_neg_correct = 0.0

    num_pos_examples = 0.0
    num_neg_examples = 0.0

    preds = []
    targets = []

    with torch.no_grad():
        for data, attr, target in loader:
            data, attr, target = data.to(device), attr.to(device), target.to(device)

            # i also need to modify the data a bit here
            data = data.float() / 255.0
            target = target.long()

            logits, probas = model(data)
            test_loss += F.cross_entropy(
                logits, target, reduction="sum"
            ).item()  # sum up batch loss
            _, pred = torch.max(probas, 1)
            num_examples += target.size(0)

            # split correctness by pos/neg examples
            num_pos_examples += target.sum()
            num_neg_examples += target.size(0) - target.sum()

            # correct should also be split
            num_pos_correct += (pred[target == 1] == target[target == 1]).sum()
            num_neg_correct += (pred[target == 0] == target[target == 0]).sum()

            preds.append(pred)
            targets.append(target)

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        preds = np.ravel(preds.data.cpu().numpy())
        targets = np.ravel(targets.data.cpu().numpy())

    test_loss /= num_examples

    # average over weighted proportions of positive/negative examples
    test_acc = (
        (num_pos_correct.float() / num_pos_examples)
        + (num_neg_correct.float() / num_neg_examples)
    ) * 0.5

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}".format(test_loss, test_acc)
    )

    return test_loss, test_acc, preds


def run_loop(model, loader, device):
    labels = []
    probs = []

    model.eval()

    with torch.no_grad():
        # iterate through entire dataset
        for data, attr, target in loader:
            data = data.float() / 255.0
            data, target = data.to(device), target.to(device).long()
            logits, probas = model(data)
            probs.append(probas)

            # save data, density ratios, and labels
            labels.append(target)
        labels = torch.cat(labels)
        probs = torch.cat(probs)
    return labels, probs


def save_checkpoint(state, is_best, epoch, folder="./", filename="checkpoint.pth.tar"):
    """Saves model checkpoint"""

    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename), os.path.join(folder, f"model_best.pth.tar")
        )


def compute_opt_nce(bias_factor, bias_ratio):
    """
    Compute Bayes Optimal NCE.
    """
    if bias_factor == "Both":
        g1_frac = (bias_ratio**2) / ((bias_ratio + 1) ** 2)
        g2_frac = g3_frac = bias_ratio / ((bias_ratio + 1) ** 2)
        g4_frac = 1 / ((bias_ratio + 1) ** 2)

        bz_g1, bz_g2, bz_g3, bz_g4 = (
            g1_frac / 0.25,
            g2_frac / 0.25,
            g3_frac / 0.25,
            g4_frac / 0.25,
        )
        pz_ref = 0.25 * (
            np.log(1 / (bz_g1 + 1))
            + np.log(1 / (bz_g2 + 1))
            + np.log(1 / (bz_g3 + 1))
            + np.log(1 / (bz_g4 + 1))
        )
        pz_bias_ratio = (
            g1_frac * np.log(bz_g1 / (bz_g1 + 1))
            + g2_frac * np.log(bz_g2 / (bz_g2 + 1))
            + g3_frac * np.log(bz_g3 / (bz_g3 + 1))
            + g4_frac * np.log(bz_g4 / (bz_g4 + 1))
        )
        opt_nce = -(pz_ref + pz_bias_ratio) / 2

    else:
        major_frac, minor_frac = bias_ratio / (1 + bias_ratio), 1 / (1 + bias_ratio)
        bz_major, bz_minor = major_frac / 0.5, minor_frac / 0.5
        pz_ref = 0.5 * (np.log(1 / (bz_minor + 1)) + np.log(1 / (bz_major + 1)))
        pz_bias_ratio = minor_frac * np.log(
            bz_minor / (bz_minor + 1)
        ) + major_frac * np.log(bz_major / (bz_major + 1))
        opt_nce = -(pz_ref + pz_bias_ratio) / 2

    return opt_nce


def show_ratio(label, z, ratio, class_label, plot=False, path=None):
    """
    Return stats of estimated importance weight w.r.t subgroups.
    Note long tails are not encouraged.
    Use in multi bias setting.
    If you want to plot, set plot to True and provide output path.
    Groups ordered as [Major class's Major subgroup, Major class's Minor subgroup,
                Minor class's Major subgroup, Minor class's Minor subgroup].

    Returns:
        Max, Mean list of subgroups
    """
    label = label.numpy()
    z = z.numpy()
    ratio = ratio.numpy()

    l_major, l_minor = class_label
    z_major, z_minor = [1, 0]

    g1 = (label == l_major) & (z == z_major)
    g2 = (label == l_major) & (z == z_minor)
    g3 = (label == l_minor) & (z == z_major)
    g4 = (label == l_minor) & (z == z_minor)

    ratio_g1 = ratio[g1]
    ratio_g2 = ratio[g2]
    ratio_g3 = ratio[g3]
    ratio_g4 = ratio[g4]

    max = []
    mean = []
    for curr_ratio in [ratio_g1, ratio_g2, ratio_g3, ratio_g4]:
        curr_max = np.max(curr_ratio)
        curr_mean = np.mean(curr_ratio)
        max.append(curr_max)
        mean.append(curr_mean)

    if plot:
        plt.hist(ratio_g1, color="r", alpha=0.6, bins=100, label=f"Clean {l_major}")
        plt.hist(
            ratio_g2, color="orange", alpha=0.6, bins=100, label=f"Rotated {l_major}"
        )
        plt.hist(ratio_g3, color="b", alpha=0.6, bins=100, label=f"Clean {l_minor}")
        plt.hist(ratio_g4, color="g", alpha=0.6, bins=100, label=f"Rotated {l_minor}")

        plt.xlabel("Importance Weights")
        plt.ylabel("Density")

        plt.legend(loc="upper right")

        plt.savefig(path)

    return max, mean


def write_data_info(label, z, digit_list, output_file):
    """
    Write data info to output file.
    Assumes output file is properly opened before.
    """

    major, minor = digit_list
    cln_major = ((label == major) & (z == 1)).sum()
    rot_major = ((label == major) & (z == 0)).sum()
    cln_minor = ((label == minor) & (z == 1)).sum()
    rot_minor = ((label == minor) & (z == 0)).sum()

    output_file.write(f"Total Dataset Size: {label.shape[0]}\n")
    output_file.write(f"\tMajor Class {major}: {cln_major+rot_major}\n")
    output_file.write(f"\t\tClean: {cln_major}\n")
    output_file.write(f"\t\tRotated: {rot_major}\n")
    output_file.write(f"\tMinor Class {minor}: {cln_minor+rot_minor}\n")
    output_file.write(f"\t\tClean: {cln_minor}\n")
    output_file.write(f"\t\tRotated: {rot_minor}\n")


# ================================================================


def main(args):
    # Config
    gpu_idx = args.gpu_idx
    batch_size = args.batch_size
    bias_ratio = args.bias_ratio
    bias_factor = args.bias_factor
    perc = args.perc
    epochs = args.epochs
    dataset = args.dataset
    lr = args.lr
    target_nce = args.target_nce
    data_dir = args.datapath
    # change with your environment if necessary
    if dataset == "mnist":
        class_label = [3, 1]
    elif dataset == "fmnist":
        class_label = [7, 1]
    else:
        NotImplementedError("Only mnist and fmnist are supported.")

    # Random Seed
    random_seed = args.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # environment
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_idx}" if use_cuda else "cpu")

    # load data
    data_dir_name = data_dir.split("/")[-1]
    print(f"Loading data: {data_dir_name}")
    data_path = os.path.join(data_dir, f"train_data.pt")
    Y_path = os.path.join(data_dir, f"train_Y.pt")
    A_path = os.path.join(data_dir, f"train_A.pt")

    # digit list
    digit_list = (
        class_label
        if bias_factor == "Y"
        else [1, 0]
        if bias_factor == "A"
        else [class_label, [1, 0]]
    )

    # build dataset
    if bias_factor in ["Y", "A"]:
        curr_label_path = A_path if bias_factor == "A" else Y_path
        balanced_train_dataset, unbalanced_train_dataset = build_dataset_single(
            data_path=data_path,
            label_path=curr_label_path,
            digit_list=digit_list,
            bias_ratio=bias_ratio,
            split="train",
            perc=perc,
        )
        balanced_valid_dataset, unbalanced_valid_dataset = build_dataset_single(
            data_path=data_path,
            label_path=curr_label_path,
            digit_list=digit_list,
            bias_ratio=bias_ratio,
            split="valid",
            perc=perc,
        )
    elif bias_factor == "Both":
        balanced_train_dataset, unbalanced_train_dataset = build_dataset_multi(
            data_path=data_path,
            label_path=Y_path,
            z_path=A_path,
            digit_list=digit_list,
            bias_ratio=bias_ratio,
            split="train",
            perc=perc,
        )
        balanced_valid_dataset, unbalanced_valid_dataset = build_dataset_multi(
            data_path=data_path,
            label_path=Y_path,
            z_path=A_path,
            digit_list=digit_list,
            bias_ratio=bias_ratio,
            split="valid",
            perc=perc,
        )
    else:
        raise NotImplementedError

    # for training the classifier
    train_dataset = BagOfDatasets([balanced_train_dataset, unbalanced_train_dataset])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # balance validation set size for proper calibration assessment
    if perc != 1.0:
        print(
            "shrinking the size of the unbalanced dataset to assess classifier calibration!"
        )
        d, g, l = shrink_valid_dataset(
            balanced_valid_dataset,
            unbalanced_valid_dataset,
            bias_factor,
            bias_ratio,
            digit_list,
        )
    else:
        d, g, l = unbalanced_valid_dataset.dataset.tensors

    adj_unbalanced_valid_dataset = TensorDataset(d, g, l)
    valid_dataset = ConcatDataset(
        [balanced_valid_dataset, adj_unbalanced_valid_dataset]
    )
    valid_loader = DataLoader(valid_dataset, 200, shuffle=False)

    # output file
    out_dir = f"./{dataset}/{data_dir_name}_perc{perc}"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fixed_dir = os.path.join(out_dir, f"seed{random_seed}_bs{batch_size}_lr{lr}")
    if not os.path.isdir(fixed_dir):
        os.makedirs(fixed_dir)
    fixed_file = open(os.path.join(fixed_dir, "results.txt"), "w")

    # prepare model
    model = CLSFR_Model().to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # start train
    best_loss = -np.inf
    print(
        f"Beginning training with random seed = {random_seed}, lr = {lr}, batch size = {batch_size}...\n"
    )
    for epoch in range(1, epochs + 1):
        train_acc, train_preds, train_loss = train(
            model, epoch, train_loader, optimizer, device
        )
        valid_loss, valid_acc, valid_preds = test(model, valid_loader, device)

        # model checkpointing
        is_best = valid_acc > best_loss
        best_loss = max(valid_acc, best_loss)
        print("epoch {}: is_best: {}".format(epoch, is_best))
        if is_best:
            best_state_dict = model.state_dict()
            best_epoch = epoch

        save_checkpoint(
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            is_best,
            epoch,
            folder=fixed_dir,
        )

    # EXTRACT BEST CLASSIFIER AND LOAD MODEL
    print(
        "Best model finished training at epoch {}: {}, loading checkpoint\n".format(
            best_epoch, best_loss
        )
    )
    model = CLSFR_Model()
    model = model.to(device)
    best_state_dict = torch.load(
        os.path.join(fixed_dir, "model_best.pth.tar")
    )["state_dict"]
    model.load_state_dict(best_state_dict)

    # Compute NCE of our model
    valid_labels, valid_probs = run_loop(model, valid_loader, device)
    y_valid = valid_labels.data.cpu().numpy()
    valid_prob_pos = valid_probs.data.cpu().numpy()

    pref, pbias = valid_prob_pos[(y_valid == 1)], valid_prob_pos[(y_valid == 0)]
    Epref, Epbias = np.log(pref[:, 1]).mean(), np.log(pbias[:, 0]).mean()
    nce_of_model = -(Epref + Epbias) / 2

    our_nce_str = f"NCE Loss for Our Classifier: {nce_of_model:.3f}"
    print(our_nce_str)

    opt_nce = compute_opt_nce(bias_factor, bias_ratio)
    opt_nce_str = f"NCE Loss for Bayes Optimal Classifier: {opt_nce:.3f}\n"
    print(opt_nce_str)

    fixed_file.write(opt_nce_str)
    fixed_file.write(our_nce_str)
    fixed_file.write("\n\n")


    # compute ratio
    my_data = torch.load(data_path)
    # (size, 28, 28) -> (size, 1, 28, 28)
    my_data = my_data.unsqueeze(1)
    my_label = torch.load(Y_path)
    my_z = torch.load(A_path)
    my_dataset = LoopingDataset(TensorDataset(my_data, my_label, my_z))
    my_train_loader = DataLoader(my_dataset.dataset, batch_size=100, shuffle=False)

    train_data = []
    train_ratios = []
    train_labels = []
    train_z = []

    # MAKE SURE YOU TURN BATCHNORM OFF!
    model.eval()

    with torch.no_grad():
        # only iterating through unbalanced dataset!
        for data, label, z in my_train_loader:
            data, label, z = data.to(device), label.to(device), z.to(device)
            data = data.float() / 255.0
            label = label.long()

            logits, probas = model(data)
            density_ratio = probas[:, 1] / probas[:, 0]

            # save data, density ratios, and labels
            train_data.append(data)
            train_ratios.append(density_ratio)
            train_labels.append(label)
            train_z.append(z)

        train_data = torch.cat(train_data)
        train_ratios = torch.cat(train_ratios)
        train_labels = torch.cat(train_labels)
        train_z = torch.cat(train_z)

    label = train_labels.data.cpu()
    z = train_z.data.cpu()
    ratio = train_ratios.data.cpu()

    # ratio plot
    # multi bias setting
    print("Saving estimation results...\n")

    if bias_factor == "Both":

        # plot ratio, write ratio stat
        max_stat, mean_stat = show_ratio(
            label,
            z,
            ratio,
            class_label,
            plot=True,
            path=os.path.join(fixed_dir, "ratio_plot.png"),
        )

        fixed_file.write("Ratio Info:\n")
        fixed_file.write("\tMax: ")
        fixed_file.write(" ".join(str(val) for val in max_stat))
        fixed_file.write("\n")
        fixed_file.write("\tMean: ")
        fixed_file.write(" ".join(str(val) for val in mean_stat))
        fixed_file.write("\n\n")

    # single bias setting
    else:
        max_stat, mean_stat = show_ratio(
            label,
            z,
            train_ratios.data.cpu(),
            class_label,
            plot=True,
            path=os.path.join(fixed_dir, "ratio_plot.png"),
        )

        fixed_file.write("Ratio Info:\n")
        fixed_file.write("\tMax: ")
        fixed_file.write(" ".join(str(val) for val in max_stat))
        fixed_file.write("\n")
        fixed_file.write("\tMean: ")
        fixed_file.write(" ".join(str(val) for val in mean_stat))
        fixed_file.write("\n\n")

    # save data in fixed mode
    train_data = (train_data * 255).to(torch.uint8)
    data = train_data.data.cpu()

    print("Saving dataset ...")
    torch.save(ratio, os.path.join(fixed_dir, f"train_ratio.pt"))
    # data saved as (size, 28, 28)
    torch.save(data.squeeze(), os.path.join(fixed_dir, f"train_data.pt"))
    torch.save(label, os.path.join(fixed_dir, f"train_Y.pt"))
    torch.save(z, os.path.join(fixed_dir, f"train_A.pt"))

    # write dataset info
    write_data_info(label, z, class_label, fixed_file)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", "-s", type=int, default=0, help="random seed")
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="batch size")
    parser.add_argument("--lr", "-lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--gpu_idx", "-gpu", type=int, default=0, help="gpu idx to run")
    parser.add_argument(
        "--bias_ratio", "-br", type=int, default=4, help="Major : Minor"
    )
    parser.add_argument(
        "--bias_factor",
        "-bf",
        type=str,
        default="A",
        help="bias factor",
        choices=["Y", "A", "Both"],
    )
    parser.add_argument(
        "--perc", "-p", type=float, default=1.0, help="Ref Data / Biased Data"
    )
    parser.add_argument("--epochs", "-e", type=int, default=8, help="number of epochs")
    parser.add_argument(
        "--dataset",
        "-data",
        type=str,
        default="mnist",
        choices=["mnist", "fmnist"],
        help=" dataset name",
    )
    parser.add_argument(
        "--target_nce",
        "-t",
        type=float,
        default=0.650,
        help="target NCE in search mode",
    )
    parser.add_argument(
        "--save_cal",
        "-c",
        action="store_true",
        default=False,
        help="if store calibration curve",
    )
    parser.add_argument(
        "--datapath", "-path", type=str, required=True, help="dataset path"
    )
    args = parser.parse_args()

    main(args)
