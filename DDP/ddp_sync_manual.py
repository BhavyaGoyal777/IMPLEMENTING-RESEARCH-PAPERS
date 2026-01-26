#BASIC TORCH IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim

#IMPORTS FOR DDP
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler 

#IMPORTS FOR DATASETS AND DATALOADERS
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler 

#SIMPLE NN FOR MNIST
class SIMPLENN(nn.Module):
    def __init__(self):
        super(SIMPLENN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@torch.no_grad() #disabling gradients for broadcasting
def broadcast_model_parameters(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0) #copying parameters from rank 0 to all other ranks


#THIS FUNCTION SUMS ALL THE GRADINETS ACROSS ALL PROCESSES AND DIVIDES BY WORLD SIZE
def all_reduce_and_divide(model,world_size):
    for param in model.parameters():
        if param.requires_grad is None:
            continue
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size


def setup(rank, world_size):                  
    os.environ["MASTER_ADDR"] = "127.0.0.1"    # (localhost for 1 machine)
    os.environ["MASTER_PORT"] = "29500"        # master port
    dist.init_process_group(                   # initialize communication backend
        backend="gloo",                        # gloo works on cpu/mac
        rank=rank,                             # this process id
        world_size=world_size                  # total number of processes
    )

def cleanup(): 
    dist.destroy_process_group()


def train(rank,world_size):
    setup(rank=rank, world_size=world_size)
    torch.manual_seed(0)

    transform = transforms.Compose([           # preprocessing pipeline
        transforms.ToTensor(),                 # convert PIL image -> torch tensor in [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # normalize MNIST with mean/std
    ])

    dataset = datasets.MNIST(                  # load MNIST dataset
        root="./data",                         # where to store data
        train=True,                            # use training split
        download=True,                         # download if missing
        transform=transform                    # apply preprocessing
    ) 

    sampler = DistributedSampler(  
        dataset,
        num_replicas=world_size,              # total number of processes
        rank=rank ,                           # this process id                                                     # sampler for distributed training
        shuffle=True                       # shuffle data
    )

    dataloader = DataLoader(                   # data loader
        dataset,
        batch_size=64,                        # samples per batch
        sampler=sampler,                                       # distributed sampler
    )


    model = SIMPLENN() # simple neural network
    broadcast_model_parameters(model) #broadcasting initial model parameters from rank 0 to all other ranks

    criterion = nn.CrossEntropyLoss()         # loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # optimizer
    model.train()                               # set model to training mode

    num_epochs = 5
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)               # set epoch for shuffling
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()              # zero gradients
            output = model(data)               # forward pass
            loss = criterion(output, target)   # compute loss
            loss.backward()                    # backward pass

            all_reduce_and_divide(model, world_size) #synchronizing gradients across all processes

            optimizer.step()                   # update parameters

            if batch_idx % 100 == 0 and rank == 0:  # print from rank 0 only
                print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")


    cleanup()                                  # clean up distributed process group



if __name__ == "__main__":
    #one can imageine this(code) running on multiple machines seprately
    world_size = 2                            # number of processes(can use multiple cores)
    mp.spawn(                                 # spawn multiple processes
        train,                                # function to be called
        args=(world_size,),                   # arguments to the function
        nprocs=world_size,                    # number of processes
        join=True                             # wait for all processes to finish
    )









