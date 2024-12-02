'''
Evaluates Utility/Fairness on Downstream task on MNIST data under binay class settings.

Config:
    gpu_num: GPU deivce number
    gen_data: path to synthetic npz data
    savedir: output folder path
    savename: output file name
    dataset: ['mnist', 'fmnist']
    model_type: ['mlp', 'cnn']
    train_mode: If True, start training. If False, skip to evaluation. 
    model_path: Target model path to evaluate fairness metric. 

Codebases:
    - Fairbatch: https://github.com/yuji-roh/fair-robust-selection/blob/main/models.py
    - GS-WGAN: https://github.com/DingfanChen/GS-WGAN.git
'''


import torch
import numpy as np
import os
import random
import json

import torchvision.transforms as tf 
import torch.utils.data as data

import torch.nn as nn
from torch import optim



# MLP / CNN Model 
class SimpleCNN(nn.Module):
    """
    Simple CNN Clssifier
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            # (N, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (N, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (N, 64, 7, 7)
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        y_ = self.conv(x) # (N, 64, 7, 7)
        y_ = y_.view(y_.size(0), -1) # (N, 64*7*7)
        y_ = self.fc(y_)
        return y_

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.output = nn.Linear(100, 2)

        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = x.view(-1, 784)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        
        return x



def get_dataloader(train_x, train_y, batch_size, num_workers):
    trainset = data.TensorDataset(train_x, train_y) 
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader

def preprocess_train_data(x_gen, y_gen, normalize, digit_list, shuffle=True):
    train_x = torch.stack([normalize(x) for x in x_gen])
    train_y = convert_to_binary(y_gen, digit_list).long()
    if shuffle:
        train_x, train_y = shuffle(train_x, train_y)

    return train_x, train_y

def convert_to_binary(label, digit_list):
    '''
    Convert given digit list to binary class (i.e. 0, 1)
    Note order matters. [digit 1, digit 2] => [1, 0]
    '''
    cls_1, cls_2 = digit_list
    mask_1 = (label == cls_1)
    mask_2 = (label == cls_2)
    label[mask_1] = 1
    label[mask_2] = 0
    return label

def shuffle(data, label, z = torch.Tensor([])):
    '''
    Shuffle given dataset.
    '''
    full_indices = np.arange(len(data))
    np.random.shuffle(full_indices)
    tensor_x = data[full_indices]
    tensor_y = label[full_indices]
    if not z.shape[0]:
        return tensor_x, tensor_y
    else:
        tensor_z = z[full_indices]
        return tensor_x, tensor_y, tensor_z

def train(model, n_epoch, train_loader, val_loader, optimizer, criterion, model_type, save_dir, seed_id, device="cpu"):
    model.train()
    prev_val_acc = 0
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(input=outputs, target=labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {}, loss = {:.3f}'.format(epoch, running_loss/len(train_loader)))

        curr_val_acc = evaluate(model, val_loader, device)
        if curr_val_acc > prev_val_acc:
            print(f'Saving Best model at epoch={epoch} with accuracy={curr_val_acc:.5f}...')
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_type}_dict_seed_{seed_id}.pkl'))
            prev_val_acc = curr_val_acc
    return 

def evaluate(model, loader, device="cpu"):
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    acc = 100*correct/total
    return acc

def predict(model, images, device):
    model = model.to(device)
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
    return predicted


# test downstream fairness
# Reference: 
def test_model(model_, X, y, s1, device):
    """Tests the performance of a model.
    Args:
        model_: A model to test.
        X: Input features of test data.
        y: True label (1-D) of test data.
        s1: Sensitive attribute (1-D) of test data.
    Returns:
        The test accuracy and the fairness metrics of the model.
    """
    
    model_.eval()
    
    # y_hat = model_(X).squeeze()
    # prediction = (y_hat > 0.0).int().squeeze()
    prediction = predict(model_, X, device)
    prediction = prediction.cpu()
    # y = (y > 0.0).int()

    z_0_mask = (s1 == 0.0)
    z_1_mask = (s1 == 1.0)
    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))
    
    y_0_mask = (y == 0.0)
    y_1_mask = (y == 1.0)
    y_0 = int(torch.sum(y_0_mask))
    y_1 = int(torch.sum(y_1_mask))
    
    Pr_y_hat_1 = float(torch.sum((prediction == 1))) / (z_0 + z_1)
    
    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1

    acc_z_0 = float(torch.sum((prediction == y)[z_0_mask])) / z_0
    acc_z_1 = float(torch.sum((prediction == y)[z_1_mask])) / z_1

        
    y_1_z_0_mask = (y == 1.0) & (s1 == 0.0)
    y_1_z_1_mask = (y == 1.0) & (s1 == 1.0)
    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))
    
    Pr_y_hat_1_y_0 = float(torch.sum((prediction == 1)[y_0_mask])) / y_0
    Pr_y_hat_1_y_1 = float(torch.sum((prediction == 1)[y_1_mask])) / y_1
    
    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1
    
    y_0_z_0_mask = (y == 0.0) & (s1 == 0.0)
    y_0_z_1_mask = (y == 0.0) & (s1 == 1.0)
    y_0_z_0 = int(torch.sum(y_0_z_0_mask))
    y_0_z_1 = int(torch.sum(y_0_z_1_mask))

    Pr_y_hat_1_y_0_z_0 = float(torch.sum((prediction == 1)[y_0_z_0_mask])) / y_0_z_0
    Pr_y_hat_1_y_0_z_1 = float(torch.sum((prediction == 1)[y_0_z_1_mask])) / y_0_z_1


    test_acc = torch.sum(prediction == y.int()).float() / len(y)

    DP = max(abs(Pr_y_hat_1_z_0 - Pr_y_hat_1), abs(Pr_y_hat_1_z_1 - Pr_y_hat_1))
    
    EO_Y_0 = max(abs(Pr_y_hat_1_y_0_z_0 - Pr_y_hat_1_y_0), abs(Pr_y_hat_1_y_0_z_1 - Pr_y_hat_1_y_0))
    EO_Y_1 = max(abs(Pr_y_hat_1_y_1_z_0 - Pr_y_hat_1_y_1), abs(Pr_y_hat_1_y_1_z_1 - Pr_y_hat_1_y_1))

    
    return {'Acc': test_acc.item(), 'Acc_Z_0': acc_z_0, 'Acc_Z_1': acc_z_1, 'DP_diff': DP, 'EO_Y0_diff': EO_Y_0, 'EO_Y1_diff': EO_Y_1, 'EqOdds_diff': max(EO_Y_0, EO_Y_1)}
# ===================================================



def main(gpu, seed, dname, num_gen_img, model_iters, lr, num_epochs, mtype, target_path, settings, test_data_path):
    # config
    gpu_num = gpu
    random_seed = seed
    dataset = dname
    num_gen_img = num_gen_img
    model_type = mtype
    target_path = target_path

    # hyperparameters
    batch_size = 128
    num_workers = 4


    # save directory
    save_name = target_path.split('/')[-2]
    save_dir = os.path.join('down_clsfc', dataset, save_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("Results saved in directory: %s" % save_dir)

    # output file
    f = open(os.path.join(save_dir, f'{model_type}_result.txt'), 'a')
    f.write(f"lr = {lr}")

    # random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # environment
    device = torch.device(f'cuda:{gpu_num}')


    # load test data
    if dataset == 'mnist':
        digit_list = [3, 1]
        test_x = torch.load(os.path.join(test_data_path, 'test_data.pt')).unsqueeze(1) / 255.0
        test_y = torch.load(os.path.join(test_data_path, 'test_Y.pt'))
        test_z = torch.load(os.path.join(test_data_path, 'test_A.pt'))

    elif dataset == 'fmnist':
        digit_list = [7, 1]
        test_x = torch.load(os.path.join(test_data_path, 'test_data.pt')).unsqueeze(1) / 255.0
        test_y = torch.load(os.path.join(test_data_path, 'test_Y.pt'))
        test_z = torch.load(os.path.join(test_data_path, 'test_A.pt'))
    else: 
        raise NotImplementedError
    

    # normalize and preprocess label
    test_mean = torch.mean(test_x)
    test_std = torch.std(test_x)
    normalize_test = tf.Normalize((test_mean,), (test_std,))
    test_x, test_y = preprocess_train_data(test_x, test_y, normalize_test, digit_list, shuffle=False)


    # prepare data loaders, models, and optimizers
    train_loaders = []
    valid_loaders = []
    models = []
    optimizers = []    


    # load generated data
    gen_data = np.load(target_path)
    x_gen = gen_data['data_x'][:20000]

    # normalize functions for data
    mean = np.mean(x_gen)
    std = np.std(x_gen)
    normalize = tf.Normalize((mean,), (std,))

    # normalize and preprocess label
    x_gen = torch.from_numpy(x_gen).view(-1, 1, 28, 28)
    y_gen = torch.from_numpy(gen_data['data_y'][:20000])
    train_x, train_y = preprocess_train_data(x_gen, y_gen, normalize, digit_list, shuffle=False)

    # train/valid split
    val_size = int(train_x.shape[0] * 0.1)
    train_x, train_y = train_x[val_size:], train_y[val_size:]
    val_x, val_y = train_x[:val_size], train_y[:val_size]

    trainloader = get_dataloader(train_x, train_y, batch_size, num_workers)
    valid_loader = get_dataloader(val_x, val_y, batch_size, num_workers)

    train_loaders.append(trainloader)   
    valid_loaders.append(valid_loader)

    # models
    model = SimpleMLP() if model_type == 'mlp' else SimpleCNN()
    models.append(model)

    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizers.append(optimizer)



    acc_list = []
    eo_list = []


    # Train and evaluate the model on each dataset
    for i, (train_loader, valid_loader) in enumerate(zip(train_loaders, valid_loaders)):
        
        # Initialize the model and optimizer
        model = models[i]
        model.to(device)
        optimizer = optimizers[i]
        
        # Define the loss function 
        criterion = nn.CrossEntropyLoss()

        # get seed id for saving model dict
        seed_id = 1

        # Train the model on the current dataset
        print('Start training...')
        train(model, num_epochs, train_loader, valid_loader, optimizer, criterion, model_type, save_dir, seed_id, device)
        print('Training Finished')

        # Evaluate the model on the test set
        output = test_model(model, test_x, test_y, test_z, device)

        # write results
        output_string = json.dumps(output)
        f.write(output_string + '\n')  

        acc_list.append(output['Acc'])
        eo_list.append(output['EqOdds_diff'])

    print('Writing results...')
    # compute stat of results
    test_acc_mean = np.mean(acc_list)
    test_acc_std = np.std(acc_list)

    test_eo_mean = np.mean(eo_list)
    test_eo_std = np.std(eo_list)

    print("test_acc_mean: ", test_acc_mean)
    print("test_eo_mean: ", test_eo_mean)

    # print mean and standard deviation
    f.write(f'==================================\n') 
    f.write('Acc:\n')
    f.write(f'\tmean: {test_acc_mean:.3f}\n')
    f.write(f'\tstd: {test_acc_std:.3f}\n')
    
    f.write('EqOdds_diff:\n')
    f.write(f'\tmean: {test_eo_mean:.3f}\n')
    f.write(f'\tstd: {test_eo_std:.3f}\n')
    f.close()

    print('Done!')



if __name__ == '__main__':

    gpu_num = 0
    seed = 1
    dname = 'mnist'
    num_gen_img = 20000
    model_iters = None
    lr = 3e-6
    num_epochs = 10
    settings = 'A'
    test_data_path = '../../dataset/mnist/original'
    result_folder_path = '...path to result folder...'
    eval_mode = 'impsir'
    
    # get gen data path list
    gen_data_path_list = []
    for x in os.listdir(result_folder_path):
        if eval_mode in x:
            for y in os.listdir(os.path.join(result_folder_path, x)):
                if '_labeled.npz' in y:
                    gen_data_path_list.append(os.path.join(result_folder_path, x, y))

    print(f"Evaluating {eval_mode}: ")

    for target_path in gen_data_path_list:
        print(f"Target Path: {target_path}")
        for mtype in ['mlp', 'cnn']:
            main(gpu_num, seed, dname, num_gen_img, model_iters, lr, num_epochs, mtype, target_path, settings, test_data_path)

