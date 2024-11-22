from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from src.models import resnet, gene, featurehook
import random
import numpy as np
import torchvision
import warnings
from src.utils.utils import *
warnings.filterwarnings('ignore')


def max_loss(x, thr):

    mask = x <= thr
    loss_above_thr = x - thr
    loss = torch.where(mask, 0, loss_above_thr)
    return loss


def distillation_loss(y, teacher_scores, T=30, alpha=1):
    """
    Compute the knowledge distillation loss between student output y and teacher output teacher_scores.
    """
    # Soft target loss (KL divergence between softened probabilities)
    soft_loss = F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1), reduction='batchmean') * (alpha * T * T)
    return soft_loss

class RunningStatsLoss(nn.Module):
    def __init__(self, teacher,alpha_p = 1 ,epsilon=1e-4, reg_strength=1e6):
        super(RunningStatsLoss, self).__init__()
        self.epsilon = epsilon
        self.reg_strength = reg_strength
        self.w = nn.ParameterList()
        self.alpha_p = alpha_p
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                param = nn.Parameter(torch.ones(module.num_features))
                self.w.append(param)

    def forward(self, means, variances, run_means, run_vars, alpha_p):
        device = means[0].device

        means_flat = torch.cat([m.flatten() for m in means]).to(device)
        variances_flat = torch.cat([v.flatten() for v in variances]).to(device)
        run_means_flat = torch.cat([rm.flatten() for rm in run_means]).to(device)
        run_vars_flat = torch.cat([rv.flatten() for rv in run_vars]).to(device)

        w_flat = torch.cat([F.softplus(w).flatten() + self.epsilon for w in self.w]).to(device)
        mean_diff = (means_flat - run_means_flat) ** 2
        std_diff = (torch.abs(variances_flat) - torch.abs(run_vars_flat)) ** 2
        wasserstein_distance = torch.sqrt(mean_diff + std_diff)

        total_loss = torch.sum(w_flat * wasserstein_distance)

        total_loss = max_loss(total_loss, thr=self.alpha_p)

        norm_1_w = torch.sum(w_flat)
        l1_regularization = self.reg_strength * (1.0 / (norm_1_w + self.epsilon))

        total_loss += l1_regularization


        return total_loss



def main(seed):

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Data-free Training')
    parser.add_argument('--seed', type=int, default= seed, metavar='S')
    parser.add_argument('--teacher_model', type=str, default=r'..\results\teacher\teacher_cifar10_95.pth')
    parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr_G', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_S', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epoch_iters', type=int, default=100)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    data_test = torchvision.datasets.CIFAR10("data/dataset" + '/', train=False, transform=transform_test,download=True)
    data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, num_workers=16)

    teacher_path = args.teacher_model
    teacher = resnet.ResNet34(num_classes=10).cuda()
    teacher.load_state_dict(torch.load(teacher_path)['state_dict'])

    student = resnet.ResNet18(num_classes=10).cuda()
    generator = gene.Generator(nz=1000).cuda()


    teacher.eval()

    loss_r_feature_layers = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(featurehook.DeepInversionFeatureHook(module))


    loss_module = RunningStatsLoss(teacher)

    optimizer_S = optim.Adam(student.parameters(), lr=args.lr_S)
    optimizer_G = optim.Adam(list(generator.parameters()) + list(loss_module.parameters()), lr=args.lr_G)
    scheduler_S = torch.optim.lr_scheduler.CyclicLR(optimizer_S, base_lr=1e-3, max_lr=6e-3, step_size_up=2000,
                                             cycle_momentum=False)
    acc_list = []


    for epoch in range(args.n_epochs):
        scheduler_S.step()

        # training step
        teacher.eval()
        student.train()
        generator.train()

        for i in range(args.epoch_iters):
            ##################
            # Student Training
            ##################

            for k in range(50):

                optimizer_S.zero_grad()

                noises = torch.randn(args.batch_size, 1000).cuda()
                fake = generator(noises).detach()
                fake,fake_for_test = Augmentation(fake)
                t_logit = teacher(fake)
                s_logit = student(fake)
                loss_DE = distillation_loss(s_logit, t_logit)

                loss_S = loss_DE
                loss_S.backward()
                optimizer_S.step()

            ####################
            # Generator Training
            ####################
            optimizer_G.zero_grad()
            noises = torch.randn(args.batch_size, 1000).cuda()
            fake = generator(noises)
            fake,fake_for_test = Augmentation(fake)

            t_logit = teacher(fake)

            means = [mod.mean.cuda() for mod in loss_r_feature_layers]
            variances = [mod.var.cuda() for mod in loss_r_feature_layers]
            run_mean=[mod.running_mean.cuda() for mod in loss_r_feature_layers]
            run_var=[mod.running_var.cuda() for mod in loss_r_feature_layers]

            loss_BNS = loss_module(means, variances, run_mean, run_var, 0.0)


            loss_G = loss_BNS.cuda()

            loss_G.backward()
            optimizer_G.step()

            if i == 0:
                print("[Epoch %d/%d] [loss_kd: %f] [loss_BNS : %f] [loss_G : %f]" % (
                    epoch, args.n_epochs, loss_DE.item(), loss_BNS.item(),loss_G.item()))

        # validataion step
        student.eval()
        generator.eval()


        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(data_test_loader):
                data, target = data.cuda(), target.cuda()
                output = student(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                noises = torch.randn(args.batch_size, args.latent_dim).cuda()
                fake = generator(noises)
                fake,extensive_augment,fake_for_test = Augmentation(fake)

                epoch_dir = './Results/DBN/generated_images/epoch{}'.format(1)
                os.makedirs(epoch_dir, exist_ok=True)
                for img_idx, image in enumerate(fake_for_test):
                    image = image.unsqueeze(0)
                    filename = os.path.join(epoch_dir, 'batch{}_img{}.png'.format(i, img_idx))
                    torchvision.utils.save_image(image, filename)


        test_loss /= len(data_test_loader.dataset)

        print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(epoch, test_loss, correct,len(data_test_loader.dataset),100. * correct / len(data_test_loader.dataset)))
        acc = correct / len(data_test_loader.dataset)
        acc_list.append(acc)

    print("Best Acc : {:.4f}".format(max(acc_list)))

    results_dir = './Results'
    os.makedirs(results_dir,exist_ok=True)

    # Construct the file path
    file_path = os.path.join(results_dir, '{}_results.txt'.format(args.dataset))

    # Open the file and write the results
    with open(file_path, 'a') as file:
        file.write(str(args))
        file.write("\nBest Acc : {:.4f}\n".format(max(acc_list)))
        file.write("Accuracy_list : " + str(acc_list) + "\n \n")

    model_save_path = './Results/DBN/generator_model.pth'
    # Save the model state dictionary

    torch.save(generator.state_dict(), model_save_path)
    model_save_path = './Results/DBN/student_model.pth'

    # Save the model state dictionary
    torch.save(student.state_dict(), model_save_path)

seed= random.randint(0,10000)
main(seed)
