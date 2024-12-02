# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pate_gan.py implements the PATE_GAN generative model to generate private synthetic data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import math
from utils.architectures import Generator, Discriminator
from utils.helper import weights_init, pate, moments_acc
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import time


def get_imp_sir_weight(idx, ratio):
    ratio_mask = (
        torch.ones_like(ratio)
        .scatter_(0, torch.Tensor(idx).long(), torch.zeros_like(ratio))
        .bool()
    )
    ratios = torch.masked_fill(ratio, ratio_mask, 0.0)
    ratio_sum = ratios.sum()
    s_i = ratio_sum - ratios
    ratio_new = ratios / s_i
    return ratio_new / ratio_new.sum()


class PATE_GAN:
    def __init__(self, input_dim, z_dim, num_teachers, target_epsilon, target_delta, conditional=True):
        self.generator = Generator(z_dim, input_dim, conditional).cuda().double()
        self.student_disc = Discriminator(input_dim, wasserstein=False).cuda().double()
        self.teacher_disc = [Discriminator(input_dim, wasserstein=False).cuda().double()
                             for _ in range(num_teachers)]
        self.generator.apply(weights_init)
        self.student_disc.apply(weights_init)
        self.z_dim = z_dim
        self.num_teachers = num_teachers
        for i in range(num_teachers):
            self.teacher_disc[i].apply(weights_init)

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.conditional = conditional

    def train(self,  x_train, y_train, hyperparams):
        batch_size = hyperparams.batch_size
        num_teacher_iters = hyperparams.num_teacher_iters
        num_student_iters = hyperparams.num_student_iters
        num_moments = hyperparams.num_moments
        lap_scale = hyperparams.lap_scale
        class_ratios = None
        if self.conditional:
            class_ratios = torch.from_numpy(hyperparams.class_ratios)

        real_label = 1
        fake_label = 0

        alpha = torch.cuda.DoubleTensor([0.0 for _ in range(num_moments)])
        l_list = 1 + torch.cuda.DoubleTensor(range(num_moments))

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=hyperparams.lr)
        optimizer_sd = optim.Adam(self.student_disc.parameters(), lr=hyperparams.lr)
        optimizer_td = [optim.Adam(self.teacher_disc[i].parameters(), lr=hyperparams.lr
                                   ) for i in range(self.num_teachers)]

        # ssoy
        z_label = x_train[:, 5]
        counts = np.unique(z_label, return_counts=True)[1]
        train_ratio = torch.Tensor(np.where(z_label==0, counts[1] / counts[0], 1))


        tensor_data = data_utils.TensorDataset(torch.cuda.DoubleTensor(x_train), \
                                               torch.cuda.DoubleTensor(y_train))

        train_loader = []
        for teacher_id in range(self.num_teachers):
            start_id = teacher_id * len(tensor_data) / self.num_teachers
            end_id = (teacher_id + 1) * len(tensor_data) / self.num_teachers if teacher_id != (
                    self.num_teachers - 1) else len(tensor_data)

            if hyperparams.resample:
                # print("Using resampling!!!!!")
                train_loader.append(data_utils.DataLoader(torch.utils.data.Subset( \
                    tensor_data, range(int(start_id), int(end_id))), \
                    batch_size=batch_size, \
                    sampler=WeightedRandomSampler(
                        train_ratio[int(start_id):int(end_id)], batch_size
                    )))
            else:
                train_loader.append(data_utils.DataLoader(torch.utils.data.Subset( \
                    tensor_data, range(int(start_id), int(end_id))), batch_size=batch_size, shuffle=True))

        steps = 0
        epsilon = 0

        while epsilon < self.target_epsilon:

            # train the teacher discriminators
            for t_2 in range(num_teacher_iters):
                for i in range(self.num_teachers):
                    inputs, categories = None, None
                    for b, data in enumerate(train_loader[i], 0):
                        inputs, categories = data
                        break

                    # train with real
                    optimizer_td[i].zero_grad()
                    label = torch.full((inputs.size()[0],), real_label).cuda()
                    output = self.teacher_disc[i].forward(torch.cat([inputs, categories.unsqueeze(1).double()], dim=1))
                    label = label.unsqueeze(1)
                    label = label.double()
                    err_d_real = criterion(output, label)
                    err_d_real.backward()

                    # train with fake
                    z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()
                    label.fill_(fake_label)

                    if self.conditional:
                        category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                        fake = self.generator(torch.cat([z.double(), category], dim=1))
                        output = self.teacher_disc[i].forward(torch.cat([fake.detach(), category], dim=1))
                    else:
                        fake = self.generator(z.double())
                        output = self.teacher_disc[i].forward(fake)

                    err_d_fake = criterion(output, label.double())
                    err_d_fake.backward()
                    optimizer_td[i].step()

            # train the student discriminator
            for t_3 in range(num_student_iters):
                z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()

                if self.conditional:
                    category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                    fake = self.generator(torch.cat([z.double(), category], dim=1))
                    predictions, clean_votes = pate(torch.cat(
                        [fake.detach(), category], dim=1), self.teacher_disc, lap_scale)
                    outputs = self.student_disc.forward(torch.cat([fake.detach(), category], dim=1))
                else:
                    fake = self.generator(z.double())
                    predictions, clean_votes = pate(fake.detach(), self.teacher_disc, lap_scale)
                    outputs = self.student_disc.forward(fake.detach())

                # update the moments
                alpha = alpha + moments_acc(self.num_teachers, clean_votes, lap_scale, l_list)

                # update student

                err_sd = criterion(outputs, predictions)

                optimizer_sd.zero_grad()
                err_sd.backward()
                optimizer_sd.step()

            # train the generator
            optimizer_g.zero_grad()
            z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()
            label = torch.full((inputs.size()[0],), real_label).cuda()

            if self.conditional:
                category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                fake = self.generator(torch.cat([z.double(), category], dim=1))
                output = self.student_disc(torch.cat([fake, category.double()], dim=1))
            else:
                fake = self.generator(z.double())
                output = self.student_disc.forward(fake)
            label = label.unsqueeze(1)
            label = label.double()
            err_g = criterion(output, label)
            err_g.backward()
            optimizer_g.step()

            # Calculate the current privacy cost
            epsilon = min((alpha - math.log(self.target_delta)) / l_list)
            if steps % 100 == 0:
                print("Step : ", steps, "Loss SD : ", err_sd.item(), "Loss G : ", err_g.item(), "Epsilon : ",
                      epsilon.item())

            steps += 1

            # if steps % 50 == 0:

            #     # generate synthetic data
            #     class_ratio_np = class_ratios.cpu().numpy()
            #     syn_data = self.generate(x_train.shape[0], class_ratio_np)
            #     X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]

            #     names = ['LR', 'Random Forest', 'Neural Network', 'GaussianNB', 'GradientBoostingClassifier']
            #     learners = []
            #     learners.append((LogisticRegression()))
            #     learners.append((RandomForestClassifier()))
            #     learners.append((MLPClassifier(early_stopping=True)))
            #     learners.append((GaussianNB()))
            #     learners.append((GradientBoostingClassifier()))

            #     print("AUC scores of downstream classifiers on test data : ")
            #     for i in range(0, len(learners)):
            #         score = learners[i].fit(X_syn, y_syn)
            #         pred_probs = learners[i].predict_proba(x_test)
            #         y_pred = learners[i].predict(x_test)
            #         auc_score = roc_auc_score(y_test, pred_probs[:, 1])
            #         # dp_score = demographic_parity_difference(y_test, y_pred, sensitive_features=z_test)
            #         # eo_score = equalized_odds_difference(y_test, y_pred, sensitive_features=z_test)
            #         positive_rate_group_0 = np.mean(y_pred[z_test == 0])
            #         positive_rate_group_1 = np.mean(y_pred[z_test == 1])
            #         dp_diff = abs(positive_rate_group_0 - positive_rate_group_1)

            #         true_positive_group_0 = np.mean(y_pred[(z_test == 0) & (y_test == 1)])
            #         true_positive_group_1 = np.mean(y_pred[(z_test == 1) & (y_test == 1)])
            #         eo_diff = abs(true_positive_group_0 - true_positive_group_1)
                    
            #         print('-' * 40)
            #         print('[{0}] AUROC: {1}'.format(names[i], auc_score))
            #         print('[{0}] Demographic Parity: {1}'.format(names[i], dp_diff))
            #         print('[{0}] Equalized Odds: {1}'.format(names[i], eo_diff))

            #         # open file to write the results
            #         with open(f'results_pategan.txt', 'a') as f:
            #             f.write('-' * 40 + '\n')
            #             f.write(f'steps: {steps}\n')
            #             f.write(f'[{names[i]}] AUROC: {auc_score}\n')
            #             f.write(f'[{names[i]}] Demographic Parity: {dp_diff}\n')
            #             f.write(f'[{names[i]}] Equalized Odds: {eo_diff}\n')


    def generate(self, num_rows, class_ratios, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim).cuda()
            if self.conditional:
                cat = torch.multinomial(class_ratios, batch_size, replacement=True).unsqueeze(1).cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)

            else:
                synthetic = self.generator(noise.double())

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps * batch_size < num_rows:
            noise = torch.randn(num_rows - steps * batch_size, self.z_dim).cuda()

            if self.conditional:
                cat = torch.multinomial(class_ratios, num_rows - steps * batch_size, replacement=True).unsqueeze(
                    1).cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)
            else:
                synthetic = self.generator(noise.double())
            synthetic_data.append(synthetic.cpu().data.numpy())

        return np.concatenate(synthetic_data)
