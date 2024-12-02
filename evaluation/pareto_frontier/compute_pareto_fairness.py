'''
Compute Pareto fairness for the generated data
'''

import torch
import numpy as np
import pandas as pd
import os

# config
dataset = 'mnist' # mnist, celeba, fmnist
dpath = '../../dataset'
gen_data_path = '...path to generated data...'
print("Dataset: ", dataset)
target_model = 'gswgan'

# load real data
if dataset == 'mnist':
    unbiased_x = torch.load(f'{dpath}/mnist/train_data.pt') / 1.0 
    unbiased_y = torch.load(f'{dpath}/mnist/train_Y.pt')
    unbiased_A = torch.load(f'{dpath}/mnist/train_A.pt')
    group_to_digit = {'minor': 1, 'major': 3}
    digit_to_group = {1: 'minor', 3: 'major'}
    
elif dataset == 'fmnist':
    unbiased_x = torch.load(f'{dpath}/fmnist/train_data.pt') / 1.0
    unbiased_y = torch.load(f'{dpath}/fmnist/train_Y.pt')
    unbiased_A = torch.load(f'{dpath}/fmnist/train_A.pt')
    group_to_digit = {'minor': 1, 'major': 7}
    digit_to_group = {1: 'minor', 7: 'major'}


# data group
cln_1 = unbiased_x[(unbiased_y == group_to_digit['minor']) & (unbiased_A == 1)]
rot_1 = unbiased_x[(unbiased_y == group_to_digit['minor']) & (unbiased_A == 0)]
cln_3 = unbiased_x[(unbiased_y == group_to_digit['major']) & (unbiased_A == 1)]
rot_3 = unbiased_x[(unbiased_y == group_to_digit['major']) & (unbiased_A == 0)]


# get centroids
centroid_dict = {}
centroid_dict['cln_minor'] = torch.mean(torch.mean(cln_1, dim = 0), dim=0)
centroid_dict['rot_minor'] = torch.mean(torch.mean(rot_1, dim = 0), dim=0)
centroid_dict['cln_major'] = torch.mean(torch.mean(cln_3, dim = 0), dim=0)
centroid_dict['rot_major'] = torch.mean(torch.mean(rot_3, dim = 0), dim=0)



# load gen data

import numpy as np
import torch

row_id = 0
eps=0
for iter_count in range(1000, 21000, 1000):

    # config ===
    gen_data_path = f'{gen_data_path}/gen_data_{iter_count}.npz'

    # ===

    gen_data_x = np.load(gen_data_path)['data_x'] * 255
    gen_data_y = np.load(gen_data_path)['data_y']   

    gen_data_x = torch.tensor(gen_data_x).float()
    gen_data_y = torch.tensor(gen_data_y).long()



    indices = np.where(gen_data_y != 0)[0]
    gen_data_x = gen_data_x[indices]
    gen_data_y = gen_data_y[indices]
    print("Y count")
    print(gen_data_y.unique(return_counts=True))


    # get z label and save data
    z = np.zeros_like(gen_data_y)


    for idx in range(0, gen_data_x.shape[0]):
        sample = torch.mean(torch.mean(gen_data_x[idx], dim=-1), 0)
        y = gen_data_y[idx]

        dist_dict = {}
        for k, v in centroid_dict.items():
            if digit_to_group[int(gen_data_y[idx].item())] not in k:
                continue
            else:
                dist = torch.dist(sample, v, 2)
                dist_dict[k] = dist

        pred = min(dist_dict, key=dist_dict.get)

        if 'cln' in pred:
            z[idx] = 1
        
    # get sens attr
    print("Z count")
    print(np.unique(z, return_counts=True))


    # save data to npz
    save_dir = os.path.dirname(gen_data_path)
    savename = os.path.basename(gen_data_path) + '_labeled'
    np.savez(os.path.join(save_dir, savename), data_z = z, data_x =gen_data_x / 255.0, data_y =gen_data_y)


    # categorize groups
    pairs = [ str(y)+str(z) for y,z in zip(gen_data_y, z)]
    print(np.unique(pairs, return_counts=True))
    groups, counts = np.unique(pairs, return_counts=True)

    # get FD
    data_distrib = torch.Tensor(counts) / len(gen_data_y)
    if len(data_distrib) !=4:
        data_distrib = torch.cat((data_distrib, torch.Tensor([0])), dim=0)
    if len(data_distrib) !=4:
        data_distrib = torch.cat((data_distrib, torch.Tensor([0])), dim=0)
    print(data_distrib)


    unif = torch.Tensor([0.25, 0.25, 0.25, 0.25])
    fd_base = torch.dist(unif, data_distrib, p=2)
    print(fd_base)

    # open csv and write it
    df = pd.read_csv("pareto.csv")

    result = [None] * 7
    result[0] = eps
    result[1] = fd_base.item()
    new_df = pd.DataFrame([result], columns=df.columns)

    row_id = row_id + 1 
    eps += 0.5
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv("pareto.csv", index=False)







        
    


