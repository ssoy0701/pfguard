'''
Compute Pareto utility for the generated data

Codebase:
    - GS-WGAN: https://github.com/DingfanChen/GS-WGAN.git
'''


import torch
import os
import numpy as np
import pandas as pd
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.fid_score import calculate_frechet_distance
import numpy as np
from collections import defaultdict



# config ==============
dataset = 'mnist' 
target_model = 'gswgan' # 'gpate' or 'datalens' or 'gswgan'
gen_data_path = '...path to generated data...'
dpath = '../../dataset/mnist/rotated/unbiased'
stat_file = '../stats/mnist/stat.npz' # e.g. './stats/mnist_stat.npz', see eval_fid for 
gpu_num = '2'
only_y = False
#========================
random_seed = 0
batch_size = 100


# random seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# environment
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

if dataset == 'mnist':
    real_data = torch.load(os.path.join(dpath, 'train_data.pt'))
    real_label = torch.load(os.path.join(dpath, 'train_Y.pt'))
    real_z = torch.load(os.path.join(dpath, 'train_A.pt'))
    # minor, major
    digit_list = [1, 3]


# prepare InceptionV3 model
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
load_model = model.cuda()



# get statistic of activation
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_act(model, batch_size, gen_data):
    '''
    Given InceptionV3 model, get statistic of gen_data. 
    Note gen_data should have size ( * , 28, 28, 1), type ndarray, and normalized from 0 to 1. (for binary)
    Returns:
        mean, cov
    '''
    model.eval()
    
    if gen_data.shape[0] < batch_size:
        print(f'Group Size({gen_data.shape[0]}) is smaller than batch size({batch_size})')
        n_batches = 1
        n_used_imgs = gen_data.shape[0]
        smaller_flag = True

    else:
        n_batches = gen_data.shape[0] // batch_size
        n_used_imgs = n_batches * batch_size
        smaller_flag = False


    pred_arr = np.empty((n_used_imgs, 2048))
    for i in tqdm(range(n_batches)):
        if smaller_flag:
            start = 0
            end = batch_size = gen_data.shape[0]
            images = gen_data[start:end]
        else:
            start = i * batch_size
            end = start + batch_size
            images = gen_data[start:end]

        if images.shape[1] != 3:
            images = images.transpose((0, 3, 1, 2))
            images = np.tile(images, [1, 3, 1, 1])

        batch = torch.from_numpy(images).type(torch.FloatTensor).cuda()
        pred = model(batch)[0]

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)

    return mu, sigma
# =======================================================




print('Loaded pre-computed statistic.')
f = np.load(stat_file)


m_real_all, s_real_all = f['mu_all'][:], f['sigma_all'][:]
m_real_cln_3, s_real_cln_3 = f['mu_cln_3'][:], f['sigma_cln_3'][:]
m_real_rot_3, s_real_rot_3 = f['mu_rot_3'][:], f['sigma_rot_3'][:]
m_real_cln_1, s_real_cln_1 = f['mu_cln_1'][:], f['sigma_cln_1'][:]
m_real_rot_1, s_real_rot_1 = f['mu_rot_1'][:], f['sigma_rot_1'][:]


# compute fid of gen_data_list
row_id = 0
for iter_count in range(9000, 21000, 1000):

    # config ===
    gen_data_path = f'{gen_data_path}/gen_data_{iter_count}.npz_labeled.npz'
    # ===

    fid_values = defaultdict(list)
    print("Current gen_data: ", gen_data_path)

    # load gen_data, gen_data_y
    if target_model == 'gpate' or target_model == 'datalens':
        gen_data = np.load(gen_data_path)

        gen_data_x = gen_data['data_x'][:60000] / 255.0
        gen_data_x = gen_data_x.reshape(-1, 28, 28, 1)
        gen_data_y = gen_data['data_y'][:60000]
        gen_data_z = gen_data['data_z'][:60000]
    
    else:
        gen_data = np.load(gen_data_path)

        gen_data_x = gen_data['data_x'][:10000] 
        gen_data_y = gen_data['data_y'][:10000]
        gen_data_z = gen_data['data_z'][:10000]

    # overall fid
    m_gen_all, s_gen_all = get_act(model, batch_size, gen_data_x)
    fid_value_all = calculate_frechet_distance(m_real_all, s_real_all, m_gen_all, s_gen_all)
    print("fid_value_all: ", fid_value_all)
    fid_values['overall'].append(np.round(fid_value_all, 3))


    minor, major = digit_list
    idx_cln_3 = (gen_data_y == major) & (gen_data_z == 1)
    idx_rot_3 = (gen_data_y == major) & (gen_data_z == 0)
    idx_cln_1 = (gen_data_y == minor) & (gen_data_z == 1)
    idx_rot_1 = (gen_data_y == minor) & (gen_data_z == 0)


    m_gen_cln_3, s_gen_cln_3 = get_act(model, batch_size, gen_data_x[idx_cln_3])
    fid_value_cln_3 = calculate_frechet_distance(m_real_cln_3, s_real_cln_3, m_gen_cln_3, s_gen_cln_3)
    
    if sum(idx_rot_3) > 0: 
        m_gen_rot_3, s_gen_rot_3 = get_act(model, batch_size, gen_data_x[idx_rot_3])
        fid_value_rot_3 = calculate_frechet_distance(m_real_rot_3, s_real_rot_3, m_gen_rot_3, s_gen_rot_3)
    
    else:
        fid_value_rot_3 = None

    if sum(idx_cln_1) > 0: 
        m_gen_cln_1, s_gen_cln_1 = get_act(model, batch_size, gen_data_x[idx_cln_1])
        fid_value_cln_1 = calculate_frechet_distance(m_real_cln_1, s_real_cln_1, m_gen_cln_1, s_gen_cln_1)
    
    else:
        fid_value_cln_1 = None
    
    if sum(idx_rot_1) > 0:
        m_gen_rot_1, s_gen_rot_1 = get_act(model, batch_size, gen_data_x[idx_rot_1])
        fid_value_rot_1 = calculate_frechet_distance(m_real_rot_1, s_real_rot_1, m_gen_rot_1, s_gen_rot_1)
    
    else:
        fid_value_rot_1 = None
        


    fid_values['fid_major_z1'].append(fid_value_cln_3)
    fid_values['fid_major_z0'].append(fid_value_rot_3)
    fid_values['fid_minor_z1'].append(fid_value_cln_1)
    fid_values['fid_minor_z0'].append(fid_value_rot_1)

    # open csv and write it
    df = pd.read_csv("pareto.csv")
    
    
    result = [None] * 7
    result[2] = fid_value_all
    result[3] = fid_value_cln_3
    result[4] = fid_value_rot_3
    result[5] = fid_value_cln_1
    result[6] = fid_value_rot_1


    new_df = pd.DataFrame([result], columns=df.columns)
    row_id = row_id + 1
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv("pareto.csv", index=False)