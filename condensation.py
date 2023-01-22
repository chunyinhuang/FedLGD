''' codes for distribution matching and gradient matching '''

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_img_mean_std_per_class(img_set, im_size, num_classes):
    means = [torch.tensor([0.0, 0.0, 0.0]) for i in range(num_classes)]
    vars = [torch.tensor([0.0, 0.0, 0.0]) for i in range(num_classes)]
    count = len(img_set) * im_size[0] * im_size[1]
    for i in range(len(img_set)):
        img, label = img_set[i]
        means[label] += img.sum(axis        = [1, 2])
        vars[label] += (img**2).sum(axis        = [1, 2])

    total_means = [mean / count for mean in means]
    total_vars  = [(var / count) - (total_mean ** 2) for (var, total_mean) in zip(vars, total_means)]
    total_stds  = [torch.sqrt(total_var) for total_var in total_vars]

    return total_means, total_stds


def get_initial_normal(train_datasets, im_size, num_classes, client_num, ipc):
    # compute means and stds
    means_, stds_ = [], []
    for train_set in train_datasets:
        mean_, std_ = compute_img_mean_std_per_class(train_set, im_size, num_classes)
        means_.append(mean_)
        stds_.append(std_)

    #initialize client images
    image_syns = []
    for idx in range(client_num):
        image_syn_classes = []
        for c in range(num_classes):
            image_syn1 = torch.normal(mean=means_[idx][c][0], std=stds_[idx][c][0], size=(ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device) # [2*50, 1, 256, 256]
            image_syn2 = torch.normal(mean=means_[idx][c][1], std=stds_[idx][c][1], size=(ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device) # [2*50, 1, 256, 256]
            image_syn3 = torch.normal(mean=means_[idx][c][2], std=stds_[idx][c][2], size=(ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device) # [2*50, 1, 256, 256]
            image_syn = torch.cat([image_syn1,image_syn2,image_syn3], dim=1).detach()
            image_syn_classes.append(image_syn)
        image_syn_classes = torch.cat(image_syn_classes, dim=0)
        image_syn_classes.requires_grad = True
        image_syns.append(image_syn_classes.to(device))
    local_label_tmp = torch.tensor(np.array([np.ones(ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    label_syns = [copy.deepcopy(local_label_tmp).to(device) for idx in range(client_num)]
    # initializa server synthetic data (10 ipcs)
    server_mean_, server_std_ = [[0, 0, 0] for c in range(num_classes)], [[0, 0, 0] for c in range(num_classes)]
    for mean_, std_ in zip(means_, stds_):
        for c in range(num_classes):
            server_mean_[c][0] += mean_[c][0]/client_num
            server_mean_[c][1] += mean_[c][1]/client_num
            server_mean_[c][2] += mean_[c][2]/client_num
            server_std_[c][0] += std_[c][0]/client_num
            server_std_[c][1] += std_[c][1]/client_num
            server_std_[c][2] += std_[c][2]/client_num
    server_image_syn = []
    for c in range(num_classes):
        image_syn1 = torch.normal(mean=server_mean_[c][0], std=server_std_[c][0], size=(ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device) 
        image_syn2 = torch.normal(mean=server_mean_[c][1], std=server_std_[c][1], size=(ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device) 
        image_syn3 = torch.normal(mean=server_mean_[c][2], std=server_std_[c][2], size=(ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device) 
        server_image_syn.append(torch.cat([image_syn1,image_syn2,image_syn3], dim=1).detach())
    server_image_syn = torch.cat(server_image_syn, dim=0)
    server_image_syn.requires_grad = True
    server_image_syn = server_image_syn.to(device)
    # server_image_syn = torch.randn(size=(num_classes*args.ipc//5, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    server_label_syn = torch.tensor(np.array([np.ones(ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    return image_syns, label_syns, server_image_syn, server_label_syn


def total_variation(x, signed_image=True):
    if signed_image:
        x = torch.abs(x)
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def l2_norm(x, signed_image=True):
    if signed_image:
        x = torch.abs(x)
    batch_size = len(x)
    loss_l2 = torch.norm(x.view(batch_size, -1), dim=1).mean()
    return loss_l2

class BNFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (input[0].permute(1, 0, 2,
                                3).contiguous().view([nch,
                                                      -1]).var(1,
                                                               unbiased=False))

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.mean = mean
        self.var = var
        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def distribution_matching_bn(image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc, image_server=None, net=None):

    lambda_sim = 0.1

    # default we use ConvNetBN
    if net == None:
        net = get_network('ConvNetBN', channel, num_classes, im_size).to(device) # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False
    

    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

    loss_avg = 0

    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    images_real_all = []
    images_syn_all = []
    batch_real = image_real[0].size(0)
    if image_server is not None:
        images_server_all = []
    for c in range(num_classes):
        img_real = image_real[c]
        img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        
        seed = int(time.time() * 1000) % 100000
        dsa_param = ParamDiffAug()
        img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
        img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)

        images_real_all.append(img_real)
        images_syn_all.append(img_syn)

        if image_server is not None:
            img_server = image_server[c]
            img_server = DiffAugment(img_server, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
            images_server_all.append()

    images_real_all = torch.cat(images_real_all, dim=0)
    images_syn_all = torch.cat(images_syn_all, dim=0)

    output_real = embed(images_real_all).detach()
    output_syn = embed(images_syn_all)

    loss += torch.sum((torch.mean(output_real.reshape(num_classes, batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, ipc, -1), dim=1))**2)

    if image_server is not None:
        images_server_all = torch.cat(images_server_all, dim=0)
        output_server = embed(images_server_all).detach()
        server_client_loss = torch.sum((torch.mean(output_server, dim=0) - torch.mean(output_syn, dim=0))**2)
        loss += lambda_sim * server_client_loss


    # # l2 and total variation loss
    # loss += lambda_sim * l2_norm(img_syn)
    # loss += lambda_sim * total_variation(img_syn)
            
    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()

    if image_server is not None:
        return loss.item(), image_syn, server_client_loss.item()
    else:
        return loss.item(), image_syn

def distribution_matching(image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc, image_server=None, net=None):

    lambda_sim = 0.5

    # default we use ConvNet
    if net == None:
        net = get_network('ConvNet', channel, num_classes, im_size).to(device) # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

    loss_avg = 0

    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        img_real = image_real[c]
        img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        
        seed = int(time.time() * 1000) % 100000
        dsa_param = ParamDiffAug()
        img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
        img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)

        output_real = embed(img_real).detach()
        output_syn = embed(img_syn)

        if image_server is not None:
            img_server = image_server[c]
            img_server = DiffAugment(img_server, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
            output_server = embed(img_server).detach()
            server_client_loss = torch.sum((torch.mean(output_server, dim=0) - torch.mean(output_syn, dim=0))**2)
            loss += lambda_sim * server_client_loss
        
        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

    # # l2 and total variation loss
    # loss += lambda_sim * l2_norm(img_syn)
    # loss += lambda_sim * total_variation(img_syn)
            
    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()

    if image_server is not None:
        return loss.item(), image_syn, server_client_loss.item()
    else:
        return loss.item(), image_syn

def gradient_matching(args, net, criterion, gw_reals, image_syn, optimizer_img, channel, num_classes, im_size, ipc):
    
    lambda_sim = 0.1

    ''' get model info '''
    net_parameters = list(net.parameters())
    
    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c

        output_syn = net(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

        loss += match_loss(gw_syn, gw_reals[c], args)

    # l2 and total variation loss
    loss += lambda_sim * l2_norm(img_syn)
    loss += lambda_sim * total_variation(img_syn)

    # # BN inversion
    # if 'BN' in args.model:
    #     loss_r_feature_layers = []
    #     for module in net.modules():
    #         if isinstance(module, torch.nn.BatchNorm2d):
    #             loss_r_feature_layers.append(BNFeatureHook(module))
    #     net(image_syn)
    #     loss_r_feature = sum([
    #         mod.r_feature
    #         for (idx, mod) in enumerate(loss_r_feature_layers)
    #     ])

    #     loss += lambda_sim * loss_r_feature

    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()

    return loss.item(), image_syn

def gradient_distribution_matching(args, net, criterion, gw_real, image_reals, image_syn, optimizer_img, channel, num_classes, im_size, ipc):
    
    lambda_sim = 0.5
    
    ''' get model info '''
    net_parameters = list(net.parameters())
    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

    # # for models contains BN
    # for module in net.modules():
    #     if 'BatchNorm' in module._get_name():  #BatchNorm
    #         module.eval() # fix mu and sigma of every BatchNorm layer

    ''' update synthetic data'''
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        
        # GM
        img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c

        output_syn = net(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

        loss += match_loss(gw_syn, gw_real, args)

        # DM
        output_syn = embed(img_syn)
        output_real = torch.zeros((ipc*5, output_syn.size(1))).to(args.device)
        for image_real in image_reals:
            img_real = image_real[c*ipc*5:(c+1)*ipc*5].reshape((ipc*5, channel, im_size[0], im_size[1]))
            output_real += embed(img_real).detach()/len(image_reals)

        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

    # l2 and total variation loss
    loss += lambda_sim * l2_norm(img_syn)
    loss += lambda_sim * total_variation(img_syn)

    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()

    return loss.item(), image_syn

def gradient_distribution_matching_bn(args, net, criterion, gw_real, image_reals, image_syn, optimizer_img, channel, num_classes, im_size, ipc):
    
    lambda_sim = 0.5
    
    ''' get model info '''
    net_parameters = list(net.parameters())
    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

    # # for models contains BN
    # for module in net.modules():
    #     if 'BatchNorm' in module._get_name():  #BatchNorm
    #         module.eval() # fix mu and sigma of every BatchNorm layer

    ''' update synthetic data'''
    loss = torch.tensor(0.0).to(device)
    images_real_all = []
    images_syn_all = []
    for c in range(num_classes):
        
        # GM
        img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c

        output_syn = net(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

        loss += match_loss(gw_syn, gw_real, args)

        # DM
        output_syn = embed(img_syn)
        output_real = torch.zeros((ipc*5, output_syn.size(1))).to(args.device)
        for image_real in image_reals:
            img_real = image_real[c*ipc*5:(c+1)*ipc*5].reshape((ipc*5, channel, im_size[0], im_size[1]))
            output_real += embed(img_real).detach()/len(image_reals)

        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

    # l2 and total variation loss
    loss += lambda_sim * l2_norm(img_syn)
    loss += lambda_sim * total_variation(img_syn)

    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()

    return loss.item(), image_syn