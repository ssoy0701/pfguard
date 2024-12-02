'''
Evaluates Utility/Fairness on Downstream task on MNIST data under multiclass settings.

Config:
    gpu_num: GPU deivce number
    gen_data: path to synthetic npz data
    savedir: output folder path
    savename: output file name
    dataset: ['mnist']
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

import torchvision.datasets as datasets
import torchvision.transforms as tf 
import torch.utils.data as data

import torch.nn as nn
from torch import optim



# MLP / CNN Model 
class SimpleCNN(nn.Module):
    """
    Simple CNN Clssifier
    """
    def __init__(self, num_classes=10):
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
        self.output = nn.Linear(100, 10)

        
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

def preprocess_train_data(x_gen, y_gen, normalize, shuffle=True):
    train_x = torch.stack([normalize(x) for x in x_gen])
    train_y = torch.nn.functional.one_hot(y_gen, num_classes=10).float()
    if shuffle:
        train_x, train_y = shuffle(train_x, train_y)

    return train_x, train_y


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

def train(model, n_epoch, train_loader, val_loader, optimizer, criterion, model_type, save_dir, mode, device="cpu"):
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
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_type}_dict_seed_{mode}.pkl'))
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
            _, label_answer = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted==label_answer).sum().item()
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
def test_model(model_, X, y, device):
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

    prediction = predict(model_, X, device)
    prediction = prediction.cpu()

    _, y = torch.max(y, 1)
    
    y_0_mask = (y == 8.0)
    y_0 = int(torch.sum(y_0_mask))

    test_acc = torch.sum(prediction == y.int()).float() / len(y)
    acc_y_0 = float(torch.sum((prediction == y)[y_0_mask])) / y_0

    
    return {'Acc': test_acc.item(), 'Acc_y_0': acc_y_0}



def main(gpu, seed, mode, dname, num_gen_img, model_iters, lr, num_epochs, mtype, target_path, test_data_path):
    # config
    gpu_num = gpu
    random_seed = seed
    dataset = dname
    num_gen_img = num_gen_img
    model_type = mtype
    target_path = target_path
    batch_size = 128
    num_workers = 4


    # save directory
    save_dir = os.path.join('multiclass')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("Results saved in directory: %s" % save_dir)



    # random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # environment
    device = torch.device(f'cuda:{gpu_num}')


    # load test data
    if dataset == 'mnist':

        testset = datasets.MNIST(root=test_data_path, train=False, download=True,
                              transform=tf.ToTensor())
        
        test_x = testset.data.unsqueeze(1).float() / 255.0
        test_y = testset.targets
    else: 
        raise NotImplementedError
    

    # normalize and preprocess label
    test_mean = torch.mean(test_x)
    test_std = torch.std(test_x)
    normalize_test = tf.Normalize((test_mean,), (test_std,))
    test_x, test_y = preprocess_train_data(test_x, test_y, normalize_test, shuffle=False)


    # prepare data loaders, models, and optimizers
    train_loaders = []
    valid_loaders = []
    models = []
    optimizers = []    

    

    # get gen data path list
    gen_data_path_list = []
    for x in sorted(os.listdir(target_path)):
        for y in os.listdir(os.path.join(target_path, x)):
            if '.npz' in y:
                gen_data_path_list.append(os.path.join(target_path, x, y))
                
    gen_data_path_list.sort(key=lambda x: int(x.split('/')[-2].split('_')[-1]))
    print(gen_data_path_list)

    for gen_path in gen_data_path_list:
        print("Evaluating: ", gen_path)
        gen_data = np.load(gen_path)
        x_gen = gen_data['data_x']

        # normalize functions for data
        mean = np.mean(x_gen)
        std = np.std(x_gen)
        normalize = tf.Normalize((mean,), (std,))

        x_gen = torch.from_numpy(x_gen).view(-1, 1, 28, 28)
        y_gen = torch.from_numpy(gen_data['data_y'])
        train_x, train_y = preprocess_train_data(x_gen, y_gen, normalize, shuffle=False)


        # train/valid split
        val_size = int(train_x.shape[0] * 0.1)
        train_x, train_y = train_x[val_size:], train_y[val_size:]
        val_x, val_y = test_x[:val_size], test_y[:val_size]

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

        # Train the model on the current dataset
        print('Start training...')
        train(model, num_epochs, train_loader, valid_loader, optimizer, criterion, model_type, save_dir, mode, device)
        print('Training Finished')

        # Evaluate the model on the test set
        model_dict = torch.load(os.path.join(save_dir, f'{model_type}_dict_seed_{mode}.pkl'))
        model.load_state_dict(model_dict)
        output = test_model(model, test_x, test_y, device)

        # write results
        print('Results: ')
        print(output)
        acc_list.append(output['Acc'])
        eo_list.append(output['Acc_y_0'])

        print('Writing results...')

    # print mean and standard deviation
    save_name = target_path.split('/')[-1]
    f = open(os.path.join(save_dir, f'{save_name}_result.txt'), 'a')
    f.write(f'==================================\n') 
    f.write(f'Acc: {acc_list}\n')
    f.write(f'Acc(y=8):{eo_list}\n')
    f.close()
    print('Done!')

    return acc_list, eo_list



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default='base')
    args = parser.parse_args()

    dname = 'mnist'
    num_gen_img = 20000
    model_iters = None
    lr = 3e-6
    num_epochs = 15
    settings = 'A'
    mtype = 'cnn'

    test_data_path = '../../dataset/mnist/original'
    target_path = '...path to result folder...'

    runs = []
    for seed in range(10):
        total_acc, minor_acc = main(args.gpu, seed, args.mode, dname, num_gen_img, model_iters, lr, num_epochs, mtype, target_path, test_data_path)
        runs.append(minor_acc)


    runs = np.array(runs)


    f = open(os.path.join('multiclass', f'{args.mode}_result.txt'), 'a')
    f.write(f'==================================\n') 
    f.write(f'Acc(y=8):\n')
    f.write(f'Mean: {np.mean(runs, axis=0)}')
    f.write(f'Std: {np.std(runs, axis=0)}')

    f.close()
    print('Done!')

