import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import progress_bar

def average_weights(client_nets):
    # Initialize the average weights dictionary
    avg_weights = {}

    # Get the state dictionary keys from the first model
    keys = client_nets[0].state_dict().keys()

    # Initialize the sum of weights with zeros for each key
    for key in keys:
        avg_weights[key] = torch.zeros_like(client_nets[0].state_dict()[key], dtype=torch.float)

    # Sum the weights from all client models
    for net in client_nets:
        for key in keys:
            avg_weights[key] += net.state_dict()[key].float()

    # Divide by the number of models to compute the average
    num_clients = len(client_nets)
    for key in keys:
        avg_weights[key] /= num_clients

    return avg_weights




def train(epoch, idx, net, trainloader, criterion, optimizer, device):
    print('\nClient Index: %d' % idx)
    print('Epoch: %d' % epoch)     
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
def test(epoch, net, testloader, criterion, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total    
    return acc


def DFRF(epoch, T, alpha, net_student, net_teacher, trainloader, optimizer, no_jsd, device):
    print('\nEpoch: %d' % epoch)
    net_student.train()
    net_teacher.eval()
    train_loss = 0
    correct = 0
    total = 0
    
    epoch_loss = 0
    distillation_epoch_loss = 0
    consistency_epoch_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader): 
        loss = 0
        if no_jsd:
            inputs_all = inputs.to(device)
            input_clean = inputs_all
        else:
            inputs_all = torch.cat(inputs, 0).to(device) 
            input_clean = inputs[0].to(device)
            
        targets = targets.to(device)
        optimizer.zero_grad()
        
        outputs = net_student(inputs_all)
        if no_jsd:
            outputs_clean = outputs
            p_clean_sof = F.softmax(outputs_clean/T, dim=1)
        else:
            outputs_clean, outputs_aug1, outputs_aug2 = torch.split(outputs, inputs[0].size(0))
            p_clean, p_aug1, p_aug2 = F.softmax(outputs_clean/T, dim=1), F.softmax(outputs_aug1/T, dim=1), F.softmax(outputs_aug2/T, dim=1)
            p_clean_sof = F.softmax(outputs_clean/T, dim=1)
        
        outputs_teacher = net_teacher(input_clean)
        p_teacher = F.softmax(outputs_teacher/T, dim=1)

        
        distillation_loss = F.kl_div(torch.clamp(p_clean_sof, 1e-7, 1.0).log(), torch.clamp(p_teacher, 1e-7, 1.0), reduction='batchmean')
        loss = distillation_loss.clone()
        #print(loss)
        #print(distillation_loss)
        
        if not no_jsd:
            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            consistency_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            loss += alpha * consistency_loss
            #print(loss)
            #print(distillation_loss)
            

        loss.backward()
        optimizer.step()
        #print(loss)
        #
        #
        #print(distillation_loss)
        
        epoch_loss += loss.item()
        distillation_epoch_loss += distillation_loss.item()
        consistency_epoch_loss += consistency_loss.item()
    
    average_loss = epoch_loss / len(trainloader)
    average_distillation_loss = distillation_epoch_loss / len(trainloader)
    average_consistency_loss = consistency_epoch_loss / len(trainloader)
    
    return average_loss, average_distillation_loss, average_consistency_loss 
        

# Training
def val(epoch, T, alpha, net_student, net_teacher, valloader, optimizer, no_jsd, device):
    print('\nEpoch: %d' % epoch)
    net_student.eval()
    net_teacher.eval()
    train_loss = 0
    correct = 0
    total = 0
    
    epoch_loss = 0
    distillation_epoch_loss = 0
    consistency_epoch_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(valloader): 
        loss = 0
        if no_jsd:
            inputs_all = inputs.to(device)
            input_clean = inputs_all
        else:
            inputs_all = torch.cat(inputs, 0).to(device) 
            input_clean = inputs[0].to(device)
            
        targets = targets.to(device)
        optimizer.zero_grad()
        
        outputs = net_student(inputs_all)
        if no_jsd:
            outputs_clean = outputs
            p_clean_sof = F.softmax(outputs_clean/T, dim=1)
        else:
            outputs_clean, outputs_aug1, outputs_aug2 = torch.split(outputs, inputs[0].size(0))
            p_clean, p_aug1, p_aug2 = F.softmax(outputs_clean/T, dim=1), F.softmax(outputs_aug1/T, dim=1), F.softmax(outputs_aug2/T, dim=1)
            p_clean_sof = F.softmax(outputs_clean/T, dim=1)
        
        outputs_teacher = net_teacher(input_clean)
        p_teacher = F.softmax(outputs_teacher/T, dim=1)

        
        distillation_loss = F.kl_div(torch.clamp(p_clean_sof, 1e-7, 1.0).log(), torch.clamp(p_teacher, 1e-7, 1.0), reduction='batchmean')
        loss = distillation_loss.clone()
        
        if not no_jsd:
            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            consistency_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            loss += alpha * consistency_loss
            

        #loss.backward()
        #optimizer.step()
        
        epoch_loss += loss.item()
        distillation_epoch_loss += distillation_loss.item()
        consistency_epoch_loss += consistency_loss.item()
    
    average_loss = epoch_loss / len(valloader)
    average_distillation_loss = distillation_epoch_loss / len(valloader)
    average_consistency_loss = consistency_epoch_loss / len(valloader)
    return average_loss, average_distillation_loss, average_consistency_loss 


def train_AM(epoch, idx, net, trainloader, criterion, optimizer, device):
    print('\nClient Index: %d' % idx)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    
    for batch_idx, (inputs, targets) in enumerate(trainloader): 
        inputs_all = torch.cat(inputs, 0).to(device) 
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs_all)
        outputs_clean, outputs_aug1, outputs_aug2 = torch.split(outputs, inputs[0].size(0))
        loss = criterion(outputs_clean, targets)
        p_clean, p_aug1, p_aug2 = F.softmax(outputs_clean, dim=1), F.softmax(outputs_aug1, dim=1), F.softmax(outputs_aug2, dim=1)
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        
    print(train_loss/(batch_idx+1))
