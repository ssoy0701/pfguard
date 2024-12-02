'''
Create class bias for the CelebA dataset.
'''

# preprocessing code for CelebA dataset adapted from @ruishu and @mhw32
import os
import torch
import tqdm
import numpy as np
from PIL import Image
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import joblib


VALID_PARTITIONS = {'train': 0, 'valid': 1, 'test': 2}
ATTR_TO_IX_DICT = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 
                    'Young': 39, 'Heavy_Makeup': 18,
                   'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 
                   'Wearing_Necktie': 38,
                   'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 
                   'Mouth_Slightly_Open': 21,
                   'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17,
                   'male': 20,
                   'Pale_Skin': 26,
                   'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 
                   'Receding_Hairline': 28, 'Straight_Hair': 32,
                   'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 
                   'Bangs': 5, 'Male': 20, 'Mustache': 22,
                   'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 
                   'Bags_Under_Eyes': 3,
                   'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 
                   'Big_Lips': 6, 'Narrow_Eyes': 23,
                   'Chubby': 13, 'Smiling': 31, 
                   'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}

# NOTE: we use all the attributes...
IX_TO_ATTR_DICT = {v:k for k,v in ATTR_TO_IX_DICT.items()}
N_ATTRS = len(ATTR_TO_IX_DICT)
N_IMAGES = 202599
ATTR_PATH = 'attributes.pt'


def preprocess_images(args):
    # automatically save outputs to "data" directory
    IMG_PATH = os.path.join(args.out_dir, '{1}_celeba_{2}_{0}x{0}.npz'.format(
        args.img_size, args.partition, args.attr))
    if args.bias > 1:
        IMG_PATH = IMG_PATH.replace('.npz', f'_bias_{args.bias}.npz')


    print('preprocessing partition {}'.format(args.partition))
    # NOTE: datasets have not yet been normalized to lie in [-1, +1]!
    transform = transforms.Compose(
        [transforms.CenterCrop(140),
        transforms.Resize(args.img_size)])
    eval_data = load_eval_partition(args.partition, args.data_dir)
    attr_data = load_attributes(eval_data, args.partition, args.data_dir)

    if os.path.exists(IMG_PATH):
        print("{} already exists.".format(IMG_PATH))
        return

    N_IMAGES = len(eval_data)
    

    data = np.zeros((N_IMAGES, args.img_size, args.img_size, 3), dtype='float')
    labels = np.zeros((N_IMAGES, 1), dtype='uint8')


    print('starting conversion...')

    for i in tqdm.tqdm(range(N_IMAGES)):
        os.path.join(
            args.data_dir, 'img_align_celeba/', '{}'.format(eval_data[i]))
        with Image.open(os.path.join(args.data_dir, 'img_align_celeba/', 
            '{}'.format(eval_data[i]))) as img:
            if transform is not None:
                img = transform(img)
        

        img = np.array(img).astype(float) / 255.
        data[i] = img

        if args.attr == 'gender':
            # male is minor group
            male_label = attr_data[i][20]
            labels[i] = 0 if male_label == 1 else 1

        elif args.attr == 'hair':
            # black is minor group
            black_label = attr_data[i][8]
            labels[i] = 0 if black_label == 1 else 1
        else:
            raise ValueError
        # =====

    # create bias
    if args.bias > 1:
        print("Creating bias for major:minor = {}:1...".format(args.bias))
        minor_idx = np.where(labels == 0)[0]
        major_idx = np.where(labels == 1)[0]


        new_cnt = major_idx.shape[0] // args.bias

        data = np.concatenate([data[minor_idx[:new_cnt]], data[major_idx]])
        labels = np.concatenate([labels[minor_idx[:new_cnt]], labels[major_idx]])

        # shuffle data
        idx = np.arange(data.shape[0])
        np.random.seed(args.seed)
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]


    np.savez(IMG_PATH, data_x=data, data_y=labels)
    label_class, label_counts = np.unique(labels, return_counts = True)
    print("label count: ", label_class, label_counts)
    print("bias level: ", label_counts[1] / label_counts[0])
    print("Saving images to {}".format(IMG_PATH))



def load_eval_partition(partition, data_dir):
    eval_data = []
    with open(os.path.join(data_dir, 'list_eval_partition.txt')) as fp:
        rows = fp.readlines()
        for row in rows:
            path, label = row.strip().split(' ')
            label = int(label)
            if label == VALID_PARTITIONS[partition]:
                eval_data.append(path)
    return eval_data



def load_attributes(paths, partition, data_dir):
    if os.path.isfile(os.path.join(data_dir, 'attr_%s.npy' % partition)):
        attr_data = np.load(os.path.join(data_dir, 'attr_%s.npy' % partition))
    else:
        attr_data = []
        with open(os.path.join(data_dir, 'list_attr_celeba.txt')) as fp:
            rows = fp.readlines()
            for ix, row in enumerate(rows[2:]):
                row = row.strip().split()
                path, attrs = row[0], row[1:]
                if path in paths:
                    attrs = np.array(attrs).astype(int)
                    attrs[attrs < 0] = 0
                    attr_data.append(attrs)
        attr_data = np.vstack(attr_data).astype(np.int64)
    attr_data = torch.from_numpy(attr_data).float()
    return attr_data



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./celebA/', type=str, 
        help='path to downloaded celebA dataset (e.g. /path/to/celeba/')
    parser.add_argument('--out_dir', default='./celebA/', type=str, 
        help='destination of outputs')
    parser.add_argument('--partition', default='train', type=str, 
        help='[train,valid,test]')
    parser.add_argument('--img_size', default=64, type=int,
        help='size of images (e.g., 28x28, 32x32...)')
    parser.add_argument('--attr', default='gender', choices=['gender', 'hair'], type=str,
        help='attribute to use as label')
    parser.add_argument('--bias', default=1, type=int,
        help='bias for major:minor (ex. 1, 2, 3, ....)')
    parser.add_argument('--seed', default=0, type=int,
        help='random seed')
    args = parser.parse_args()
    preprocess_images(args)