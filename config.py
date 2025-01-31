import argparse
import os
import sys

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.init_args()
        self.parse()
        self.print_args()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def init_args(self):
        #Run Args
        self.parser.add_argument('--run_name', default='run', type=str, help='run name')
        self.parser.add_argument('--device_idx', default=0, type=int, help='GPU ID')
        self.parser.add_argument('--seed', default=0, type=int, help='seed')
        self.parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--num_clients', default=4, type=int, help="Number of clients")
        
        #Optimizer Args
        self.parser.add_argument('--client_lr', default=0.1, type=float, help='learning rate')
        self.parser.add_argument('--server_lr', default=0.001, type=float, help='learning rate')
        
        #Model Args
        self.parser.add_argument('--net_type', default="VGG16", type=str, help='network type')
        
        #Dataset Args
        self.parser.add_argument('--client_dataset', default="CIFAR10", type=str, help='client dataset')
        self.parser.add_argument('--server_dataset', default="CIFAR100", type=str, help='client dataset')
        
        #Augmix Args
        self.parser.add_argument('--mixture_width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
        self.parser.add_argument('--mixture_depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
        self.parser.add_argument('--aug_severity', default=3, type=int, help='Severity of base augmentation operators')
        self.parser.add_argument('--no_jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
        self.parser.add_argument('--all_ops', '-all', action='store_true', help='Turn on all operations (+brightness,contrast,color,sharpness).')
        
        #DFRF Args
        self.parser.add_argument('--T', default=1, type=int, help='Knowledge distillation temperature')
        self.parser.add_argument('--alpha', default=1, type=int, help='alpha loss hyperparameter')
        self.parser.add_argument('--patience', default=5, type=int, help='DFRF early stopping patience')
        
        #Training Args
        self.parser.add_argument('--epochs', default = 200, type=int, help='Total number of client epochs')
        self.parser.add_argument('--DFRF_epochs', default = 200, type=int, help='Max number of DFRF epochs')
        self.parser.add_argument('--DFRF_period', default = 200, type=int, help="Number of clean epochs per DFRF")
        
        #Testing Args
        
        
        
        
    def parse(self):
        args = self.parser.parse_args()
        
        #Run Args
        self.run_name = args.run_name
        self.device_idx = args.device_idx
        self.seed = args.seed
        self.resume = args.resume
        self.num_clients = args.num_clients
        
        #Optimizer Args
        self.client_lr = args.client_lr
        self.server_lr = args.server_lr
        
        #Model Args
        self.net_type = args.net_type
        
        #Dataset Args
        self.client_dataset = args.client_dataset
        self.server_dataset = args.server_dataset
        
        #Augmix Args
        self.mixture_width = args.mixture_width
        self.mixture_depth = args.mixture_depth
        self.aug_severity = args.aug_severity
        self.no_jsd = args.no_jsd
        self.all_ops = args.all_ops
        
        #DFRF Args
        self.T = args.T
        self.alpha = args.alpha
        self.patience = args.patience
        
        #Training Args
        self.epochs = args.epochs
        self.DFRF_epochs = args.DFRF_epochs
        self.DFRF_period = args.DFRF_period
        
        #Testing Args
    
    def print_args(self):
        directory = self.run_name
        
        if os.path.exists(directory):
            print(f"Error: Directory '{directory}' already exists.")
            sys.exit(1)  # Exit the program with a non-zero exit code
            
        os.makedirs(directory)
        
        file_path = os.path.join(directory, "config.txt")
        
        with open(file_path, 'w') as file:
            line = "Run Arguments"
            print(line)
            file.write(line+"\n")
            for key, value in vars(self).items():
                if not key.startswith('_') and key != 'parser':  # Exclude private attributes and the parser itself
                    line = f"{key}: {value}"
                    print(line)
                    file.write(line+"\n")
        
        
        
        
        
        
        
