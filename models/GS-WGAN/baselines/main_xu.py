import os, sys
import numpy as np
import random
import copy
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from collections import defaultdict

from config import *
from models import *
from utils import *

from ops import exp_mov_avg

IMG_DIM = 784
NUM_CLASSES = 10
CLIP_BOUND = 1.0
SENSITIVITY = 2.0



def get_sir_weight(idx, ratio):
    ratio_mask = torch.ones_like(ratio).scatter_(0, torch.Tensor(idx).cpu().long(), torch.zeros_like(ratio)).bool()
    ratios = torch.masked_fill(ratio, ratio_mask, 0.)
    ratio_sum = ratios.sum()
    return ratios / ratio_sum


def get_imp_sir_weight(idx, ratio):
    ratio_mask = (
        torch.ones_like(ratio)
        .scatter_(0, torch.Tensor(idx).cpu().long(), torch.zeros_like(ratio))
        .bool()
    )
    ratios = torch.masked_fill(ratio, ratio_mask, 0.0)
    ratio_sum = ratios.sum()
    s_i = ratio_sum - ratios
    ratio_new = ratios / s_i
    return ratio_new / ratio_new.sum()


##########################################################
### hook functions
##########################################################
def master_hook_adder(module, grad_input, grad_output):
    """
    global hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    """
    dummy hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    pass


def modify_gradnorm_conv_hook(module, grad_input, grad_output):
    """
    gradient modification hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    ### get grad wrt. input (image)

    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP




    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


def count_exceeding_gradients(class_labels, gradients, threshold):
    # Dictionary to store count of gradients exceeding threshold for each class
    gradient_bigger = defaultdict(int)
    gradient_smaller = defaultdict(int)

    # Iterate through each class label and corresponding gradient
    for class_label, gradient in zip(class_labels, gradients):
        if gradient > threshold:
            gradient_bigger[class_label] += 1
        else:
            gradient_smaller[class_label] += 1

    return gradient_bigger, gradient_smaller


def dp_conv_hook(module, grad_input, grad_output):
    """
    gradient modification + noise hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    """
    global noise_multiplier


    global c_dict
    global batch_class

    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize


    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)


    # count exceeding gradients
    gradient_bigger, gradient_smaller = count_exceeding_gradients(batch_class, grad_input_norm, clip_bound_)
    
    mo_list = [[gradient_bigger[i], gradient_smaller[i]] for i in range(2)]

    mo_list = torch.tensor(mo_list, dtype=torch.float32).reshape(2, 2)

    # add noise with sensitivity 10*noise_multiplier
    mo_noise = (3.16 * noise_multiplier * torch.randn_like(mo_list))
    noisy_mo_list = mo_list + mo_noise  
    noisy_m = noisy_mo_list[:, 0].sum()

    C_k_list = []
    for k in range(2):
        
        noisy_bk = noisy_mo_list[k].sum()

        if noisy_bk <= 0:
            adjust_ratio_k = 0
        elif noisy_m == 0:
            adjust_ratio_k = 0
        else:
            adjust_ratio_k =  ( noisy_mo_list[k, 1] / noisy_bk) / (noisy_m / batchsize)

        C_k =  (1 + adjust_ratio_k)
        
        c_dict[k].append(C_k) # for log

        # limit xu
        if C_k > 1.5:
            C_k = 1.5
        elif C_k < 1:
            C_k = 1
        C_k_list.append(C_k) # for this batch



    # update clip_coef    
    batch_ck_list = []
    for i in batch_class:
        batch_ck_list.append(C_k_list[i])

    clip_coef = torch.mul(clip_coef, torch.tensor(batch_ck_list))
    max_ck = torch.tensor(batch_ck_list).max()

    global max_c
    max_c = max_ck.item()



    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)

    grad_wrt_image = clip_coef * grad_wrt_image




    ### add noise
    noise = (
        max_c * clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    )
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


##########################################################
### main
##########################################################
def main(args):
    ### config
    global noise_multiplier

    # ssoy ====
    global c_dict
    global batch_class
    global max_c
    
    # c for groups
    c_dict = {0: [], 1: []}
    # ====


    dataset = args.dataset
    num_discriminators = args.num_discriminators
    noise_multiplier = args.noise_multiplier
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    L_epsilon = args.L_epsilon
    critic_iters = args.critic_iters
    latent_type = args.latent_type
    load_dir = args.load_dir
    save_dir = args.save_dir
    if_dp = noise_multiplier > 0.0
    gen_arch = args.gen_arch
    num_gpus = args.num_gpus
    uncond = args.uncond

    max_c = []

    ### CUDA
    use_cuda = torch.cuda.is_available()
    devices = [torch.device(f"cuda:{args.target_gpu}" if use_cuda else "cpu")]
    device0 = devices[0]
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    ### Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ### Fix noise for visualization
    if latent_type == "normal":
        fix_noise = torch.randn(50, z_dim)
    elif latent_type == "bernoulli":
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((50, z_dim)).view(50, z_dim)
    else:
        raise NotImplementedError

    ### Set up models
    if gen_arch == "DCGAN":
        netG = GeneratorDCGAN(z_dim=z_dim, model_dim=model_dim, num_classes=10)
    elif gen_arch == "ResNet":
        netG = GeneratorResNet(z_dim=z_dim, model_dim=model_dim, num_classes=10)
    else:
        raise ValueError

    netGS = copy.deepcopy(netG).to(device0)
    netD_list = []
    for i in range(num_discriminators):
        netD = DiscriminatorDCGAN()
        netD_list.append(netD)

    ### Load pre-trained discriminators
    if load_dir is not None:
        for netD_id in range(num_discriminators):
            print("Load NetD ", str(netD_id))
            network_path = os.path.join(load_dir, "netD_%d" % netD_id, "netD.pth")
            netD = netD_list[netD_id]
            netD.load_state_dict(torch.load(network_path, map_location="cpu"))
            netD.to(device0)    

    netG = netG.to(device0)
    # for netD_id, netD in enumerate(netD_list):
    #     device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
    #     netD.to(device)

    ### Set up optimizers
    optimizerD_list = []
    for i in range(num_discriminators):
        netD = netD_list[i]
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD_list.append(optimizerD)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # if uncond:
    #     digit_class = [0, 0]
    # elif dataset == "mnist":
    #     digit_class = [1, 3]
    # elif dataset == "fmnist":
    #     digit_class = [1, 7]
    # else:
    #     raise NotImplementedError

    digit_class = [1, 3]

    img_datapath = os.path.join(args.datapath, "train_data.pt")
    train_data = torch.load(img_datapath, map_location="cpu").unsqueeze(1).float() / 255

    label_datapath = os.path.join(args.datapath, "train_Y_combined.pt")
    if uncond:
        train_label = torch.zeros(train_data.size()[0], dtype=torch.long, device="cpu")
    else:
        train_label = torch.load(label_datapath, map_location="cpu").long()

    # # we assume groups are known
    # # ratio_datapath = os.path.join(args.datapath, "train_ratio.pt")
    # ratio_datapath = os.path.join(args.datapath, "train_ratio_answer.pt")

    # if args.resample:
    #     train_ratio = torch.load(ratio_datapath, map_location="cpu")
    # else:
    train_ratio = torch.ones_like(train_label)

    trainset = data.TensorDataset(train_data, train_label)

    if load_dir is not None:
        assert os.path.exists(os.path.join(load_dir, "indices.npy"))
        print("load indices from disk")
        indices_full = np.load(os.path.join(load_dir, "indices.npy"), allow_pickle=True)
    else:
        print("create indices file")
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        indices_full.dump(os.path.join(save_dir, "indices.npy"))
    trainset_size = int(len(trainset) / num_discriminators)
    print("Size of the dataset: ", trainset_size)

    input_pipelines = []
    for i in range(num_discriminators):
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = data.DataLoader(
            trainset,
            batch_size=batchsize,
            drop_last=False,
            num_workers=args.num_workers,
            sampler=WeightedRandomSampler(
                get_imp_sir_weight(indices, train_ratio), batchsize
            ),
        )
        input_data = inf_train_gen(trainloader)
        input_pipelines.append(input_data)

    ### Register hook
    global dynamic_hook_function
    for netD in netD_list:
        netD.conv1.register_backward_hook(master_hook_adder)

    for iter in range(args.iterations + 1):
        #########################
        ### Update D network
        #########################
        netD_id = np.random.randint(num_discriminators, size=1)[0]
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD = netD_list[netD_id]
        optimizerD = optimizerD_list[netD_id]
        input_data = input_pipelines[netD_id]

        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(critic_iters):
            real_data, real_y = next(input_data)
            real_data = real_data.view(-1, IMG_DIM)
            real_data = real_data.to(device)
            real_y = real_y.to(device)
            real_data_v = autograd.Variable(real_data)

            ### train with real
            dynamic_hook_function = dummy_hook
            netD.zero_grad()
            D_real_score = netD(real_data_v, real_y)
            D_real = -D_real_score.mean()

            ### train with fake
            batchsize = real_data.shape[0]
            if latent_type == "normal":
                noise = torch.randn(batchsize, z_dim).to(device0)
            elif latent_type == "bernoulli":
                noise = (
                    bernoulli.sample((batchsize, z_dim))
                    .view(batchsize, z_dim)
                    .to(device0)
                )
            else:
                raise NotImplementedError
            noisev = autograd.Variable(noise)
            fake = autograd.Variable(netG(noisev, real_y.to(device0)).data)
            inputv = fake.to(device)
            D_fake = netD(inputv, real_y.to(device))
            D_fake = D_fake.mean()

            ### train with gradient penalty
            gradient_penalty = netD.calc_gradient_penalty(
                real_data_v.data, fake.data, real_y, L_gp, device
            )
            D_cost = D_fake + D_real + gradient_penalty

            ### train with epsilon penalty
            logit_cost = L_epsilon * torch.pow(D_real_score, 2).mean()
            D_cost += logit_cost

            ### update
            D_cost.backward()
            Wasserstein_D = -D_real - D_fake
            optimizerD.step()

        del (
            real_data,
            real_y,
            fake,
            noise,
            inputv,
            D_real,
            D_fake,
            logit_cost,
            gradient_penalty,
        )
        torch.cuda.empty_cache()

        ############################
        # Update G network
        ###########################
        if if_dp:
            ### Sanitize the gradients passed to the Generator
            dynamic_hook_function = dp_conv_hook
        else:
            ### Only modify the gradient norm, without adding noise
            dynamic_hook_function = modify_gradnorm_conv_hook

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        ### train with sanitized discriminator output
        if latent_type == "normal":
            noise = torch.randn(batchsize, z_dim).to(device0)
        elif latent_type == "bernoulli":
            noise = (
                bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device0)
            )
        else:
            raise NotImplementedError


        rand_index = torch.randint(0, 2, [batchsize], device="cpu")
        batch_class = rand_index.tolist()

        label = torch.LongTensor(digit_class)[
           rand_index
        ].to(device0)
        noisev = autograd.Variable(noise)
        fake = netG(noisev, label)
        fake = fake.to(device)
        label = label.to(device)
        G = netD(fake, label)
        G = -G.mean()

        ### update
        G.backward()
        G_cost = G
        optimizerG.step()

        ### update the exponential moving average
        exp_mov_avg(netGS, netG, alpha=0.999, global_step=iter)

        ############################
        ### Results visualization
        ############################
        if iter < 5 or iter % args.print_step == 0:
            print(
                "G_cost:{:5f}, D_cost:{:5f}, Wasserstein:{:5f}".format(
                    G_cost.cpu().data, D_cost.cpu().data, Wasserstein_D.cpu().data
                )
            )

            # # open file and log max_c
            # with open(os.path.join(save_dir, "max_c.txt"), "a") as f:
            #     f.write(str(max_c) + ", ")


        if iter % args.vis_step == 0:
            generate_image(
                iter, netGS, fix_noise, save_dir, device0, digit_class, num_classes=10
            )

        if iter > 0 and iter % args.save_step == 0:
            ### save model
            torch.save(
                netGS.state_dict(), os.path.join(save_dir, "netGS_%d.pth" % iter)
            )

        del label, fake, noisev, noise, G, G_cost, D_cost
        torch.cuda.empty_cache()

    ### save model
    # torch.save(netG.state_dict(), os.path.join(save_dir, 'netG.pth'))
    torch.save(netGS.state_dict(), os.path.join(save_dir, "netGS.pth"))
    np.savez(os.path.join(save_dir, "c_dict.npz"), data=c_dict)

    ### save generate samples
    save_gen_data(os.path.join(save_dir, 'gen_data.npz'), netGS, z_dim, device0, digit_class, latent_type=latent_type)


if __name__ == "__main__":
    args = parse_arguments()
    save_config(args)
    main(args)
