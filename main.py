import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

import random
import numpy as np

from models import *
from utils import progress_bar

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import augmentations
import pynvml
from config import Config
from dataset import *
from model_ops import *
import copy


config = Config()

#SEED
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)  # if using multiple GPUs
torch.backends.cudnn.deterministic = True

device = f'cuda:{config.device_idx}' if torch.cuda.is_available() else 'cpu'

pynvml.nvmlInit()
handle =pynvml.nvmlDeviceGetHandleByIndex(config.device_idx)

client_dataset, server_dataset, test_dataset, corrupt_test_dataset, val_dataset = get_dataset(config.client_dataset, config.server_dataset, config.mixture_width, config.mixture_depth, config.aug_severity, config.no_jsd, config.all_ops)

client_datasets = split_dataset(client_dataset, config.num_clients)

client_loaders, server_loader, test_loader, corrupt_test_loader, val_loader = get_loaders(client_datasets, server_dataset, test_dataset, corrupt_test_dataset, val_dataset)

if ((config.client_dataset == "CIFAR10") or (config.client_dataset == "AUGMIX_CIFAR10")):
    num_classes = 10
elif ((config.client_dataset == "CIFAR100") or (config.client_dataset == "AUGMIX_CIFAR100")):
    num_classes = 100

print("Number of classes = "+str(num_classes))

if config.net_type == "VGG16":
    client_nets = [VGG('VGG16', num_classes) for i in range(config.num_clients)]
    server_net = VGG('VGG16', num_classes)
elif config.net_type == "ResNet18":
    client_nets = [ResNet18(num_classes) for i in range(config.num_clients)]
    server_net = ResNet18(num_classes)
elif config.net_type == "MobileNetV2":
    client_nets = [MobileNetV2(num_classes) for i in range(config.num_clients)]
    server_net = MobileNetV2(num_classes)
    
for i in range(config.num_clients):
    client_nets[i].load_state_dict(server_net.state_dict())
    client_nets[i].to(device)
server_net.to(device)


client_criterions = [nn.CrossEntropyLoss() for i in range(config.num_clients)]
client_optimizers = [optim.SGD(client_nets[i].parameters(), lr=config.client_lr, momentum=0.9, weight_decay=5e-4) for i in range(config.num_clients)]
client_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(client_optimizers[i], T_max=config.epochs) for i in range(config.num_clients)]

server_criterion = nn.CrossEntropyLoss()
#server_optimizer = optim.SGD(server_net.parameters(), lr=config.server_lr, momentum=0.9, weight_decay=5e-4) 
#server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=config.DFRF_epochs)



acc_array = np.zeros(config.epochs)
rob_acc_array = np.zeros(config.epochs)
preDFRF_acc_array = np.zeros(config.epochs)
preDFRF_rob_acc_array = np.zeros(config.epochs)
time_array = np.zeros((config.epochs, config.num_clients))
energy_array = np.zeros((config.epochs, config.num_clients))


for epoch in range(config.epochs):
    
    #Client Training
    for i in range(config.num_clients):
        start_time = time.time()
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        if ((config.client_dataset != "AUGMIX_CIFAR10") and (config.client_dataset != "AUGMIX_CIFAR100")):
            train(epoch, i, client_nets[i], client_loaders[i], client_criterions[i], client_optimizers[i], device)
        else:
            train_AM(epoch, i, client_nets[i], client_loaders[i], client_criterions[i], client_optimizers[i], device)
        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        end_time = time.time()
        
        if epoch != 0:
            time_array[epoch, i] = end_time - start_time + time_array[epoch-1, i]
            energy_array[epoch, i] = end_energy - start_energy + energy_array[epoch-1, i]
        else:
            time_array[epoch, i] = end_time - start_time
            energy_array[epoch, i] = end_energy - start_energy
        client_schedulers[i].step()
    
    #Model aggregation
    avg_weights = average_weights(client_nets)
    server_net.load_state_dict(avg_weights)
    
    #Robustification
    if ((epoch+1) >= -1) and ((epoch+1) % config.DFRF_period == 0):
        print("STARTING DFRF AT EPOCH: "+str(epoch))
        
        pre_DFRF_state = {
            'net': server_net.state_dict(),
            'epoch': epoch,
        }
        torch.save(pre_DFRF_state, config.run_name+'/checkpoint_preDFRF_'+str(epoch)+'.pth')
        preDFRF_test_acc = test(epoch, server_net, test_loader, server_criterion, device)
        preDFRF_rob_test_acc = test(epoch, server_net, corrupt_test_loader, server_criterion, device)
        
        print("Pre-DFRF Clean Accuracy : "+str(preDFRF_test_acc))
        print("Pre-DFRF Robust Accuracy : "+str(preDFRF_rob_test_acc))
        
        preDFRF_acc_array[epoch] = preDFRF_test_acc
        preDFRF_rob_acc_array[epoch] = preDFRF_rob_test_acc
        
        #Initialize teacher and students
        net_teacher = copy.deepcopy(server_net)
        net_student = copy.deepcopy(server_net)
        
        #Move to GPU
        net_teacher = net_teacher.to(device)
        net_student = net_student.to(device)
        
        DFRF_criterion = nn.CrossEntropyLoss()
        DFRF_optimizer = optim.SGD(net_student.parameters(), lr=config.server_lr, momentum=0.9, weight_decay=5e-4)
        DFRF_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(DFRF_optimizer, T_max=config.DFRF_epochs)
        
        best_val_loss = float('inf')
        no_improvement_count = 0
        
        loss_array = np.zeros(config.DFRF_epochs)
        distillation_loss_array = np.zeros(config.DFRF_epochs)
        consistency_loss_array = np.zeros(config.DFRF_epochs)
        val_loss_array = np.zeros(config.DFRF_epochs)
        val_distillation_loss_array = np.zeros(config.DFRF_epochs)
        val_consistency_loss_array = np.zeros(config.DFRF_epochs)
        DFRF_acc_array = np.zeros(config.DFRF_epochs)
        DFRF_rob_acc_array = np.zeros(config.DFRF_epochs)
        
        for DFRF_epoch in range(config.DFRF_epochs):
            print("\nDFRF EPOCH: "+str(DFRF_epoch))
            
            t1 = time.time()
            loss, distillation_loss, consistency_loss  = DFRF(epoch, config.T, config.alpha, net_student, net_teacher, server_loader, DFRF_optimizer, config.no_jsd, device)
            t2 = time.time()
            print("TIME="+str(t2-t1))
            print(loss)
            print(distillation_loss)
            print(consistency_loss)
            
            loss_array[DFRF_epoch] = loss
            distillation_loss_array[DFRF_epoch] = distillation_loss
            consistency_loss_array[DFRF_epoch] = consistency_loss
            
            val_loss, val_distillation_loss, val_consistency_loss  = val(epoch, config.T, config.alpha, net_student, net_teacher, val_loader, DFRF_optimizer, config.no_jsd, device)
            print(val_loss)
            print(val_distillation_loss)
            print(val_consistency_loss)
            
            val_loss_array[DFRF_epoch] = val_loss
            val_distillation_loss_array[DFRF_epoch] = val_distillation_loss
            val_consistency_loss_array[DFRF_epoch] = val_consistency_loss
            
            print("\nDFRF CLEAN TESTING")
            DFRF_acc = test(epoch, net_student, test_loader, DFRF_criterion, device)
            print(DFRF_acc)
            
            #if(DFRF_epoch+1) % 5 == 0:
            if epoch == 199:
                print("\nDFRF ROBUST TESTING")
                DFRF_rob_test_acc = test(epoch, net_student, corrupt_test_loader, server_criterion, device)
                DFRF_rob_acc_array[DFRF_epoch] = DFRF_rob_test_acc
            
            DFRF_acc_array[DFRF_epoch] = DFRF_acc
            
            # Check if validation loss improved
            if val_loss < best_val_loss:
            #if epoch == 199:
                best_val_loss = val_loss
                no_improvement_count = 0  # Reset the counter
                print('Validation loss improved. Saving model...')
                
                # Save the best model
                state = {
                    'net': net_student.state_dict(),
                    'acc': DFRF_acc,
                    'epoch': DFRF_epoch,
                }
                torch.save(state, config.run_name+"/checkpoint_robust_"+str(epoch)+'.pth')
                
            else:
                #No improvement in validation loss
                no_improvement_count += 1
                
            #Early Stopping Condition
            if no_improvement_count >= config.patience:
                print(f"Early stopping at epoch {epoch} due to no improvement for {config.patience} epochs.")
                break
            
            
            
            print(f"Validation Loss: {val_loss}, Early Stopping Counter: {no_improvement_count}, Test Accuracy: {DFRF_acc}")
            DFRF_scheduler.step()
        
        #Save arrays
        np.save(config.run_name+'/DFRF_accuracy_'+str(epoch)+'.npy', DFRF_acc_array)
        np.save(config.run_name+'/DFRF_rob_accuracy_'+str(epoch)+'.npy', DFRF_rob_acc_array)
        np.save(config.run_name+'/DFRF_loss_'+str(epoch)+'.npy', loss_array)
        np.save(config.run_name+'/DFRF_distloss_'+str(epoch)+'.npy', distillation_loss_array)
        np.save(config.run_name+'/DFRF_constloss_'+str(epoch)+'.npy', consistency_loss_array)
        np.save(config.run_name+'/DFRF_valloss_'+str(epoch)+'.npy', val_loss_array)
        np.save(config.run_name+'/DFRF_valdistloss_'+str(epoch)+'.npy', val_distillation_loss_array)
        np.save(config.run_name+'/DFRF_valconstloss_'+str(epoch)+'.npy', val_consistency_loss_array)
        
        #Robust net is now new server_net
        robust_net = torch.load(config.run_name+"/checkpoint_robust_"+str(epoch)+'.pth', map_location=device)
        server_net.load_state_dict(robust_net['net'])
    
    #Model Testing
    print("\nCLEAN TESTING")
    test_acc = test(epoch, server_net, test_loader, server_criterion, device)
    acc_array[epoch] = test_acc
    
    if((epoch+1) % 200 == 0):
        print("\nROBUST TESTING")
        rob_test_acc = test(epoch, server_net, corrupt_test_loader, server_criterion, device)
        rob_acc_array[epoch] = rob_test_acc
    
    #Model saving
    if (epoch >= 49 and (epoch+1) % 10 == 0):
        state = {
            'net': server_net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, config.run_name+'/checkpoint_'+str(epoch)+'.pth')
    
    
    #Model sent back to clients
    for i in range(config.num_clients):
        client_nets[i].load_state_dict(server_net.state_dict())
        
    
        
np.save(config.run_name+'/clean_accuracy.npy', acc_array)
np.save(config.run_name+'/rob_accuracy.npy', rob_acc_array)
np.save(config.run_name+'/preDFRF_clean_accuracy.npy', preDFRF_acc_array)
np.save(config.run_name+'/preDFRF_rob_accuracy.npy', preDFRF_rob_acc_array)
np.save(config.run_name + '/training_time.npy', time_array)
np.save(config.run_name + '/training_energy.npy', energy_array)

with open(config.run_name+'/summary.txt', 'w') as file:
    line = "Run Summary"
    print(line)
    file.write(line+"\n")
    line = "Final Clean Accuracy = "+str(acc_array[-1])
    print(line)
    file.write(line+"\n")
    line = "Final Robust Accuracy = "+str(rob_acc_array[-1])
    print(line)
    file.write(line+"\n")
    line = "Average Client Time = "+str(np.mean(time_array[-1, :]))
    print(line)
    file.write(line+"\n")
    line = "Average Client Energy = "+str(np.mean(energy_array[-1, :]))
    print(line)
    file.write(line+"\n")
    file.write(",".join(map(str, preDFRF_acc_array)) + "\n")
    file.write(",".join(map(str, preDFRF_rob_acc_array)) + "\n")
        
