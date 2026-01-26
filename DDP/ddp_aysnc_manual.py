import os                                        
import time                                    
from contextlib import contextmanager           

import torch                                     
import torch.nn as nn                            
import torch.optim as optim                    
import torch.distributed as dist                
import torch.multiprocessing as mp              
from torch import Tensor                        


class DDP(nn.Module):                            
    def __init__(self, module):                  
        super().__init__()                       
        self.module = module                   

        # broadcast weights ---------------------------------------------------
        # goal: make sure all ranks start with exactly the same parameters
        if dist.is_initialized():                # check if process group exists
            for p in self.module.parameters():   # loop over model parameters
                
                dist.broadcast(p.data, src=0)    # copy rank0's param tensor into all ranks

        # register hooks ------------------------------------------------------
        # hooks run automatically during backward when a param gradient is ready
        for p in self.module.parameters():       # loop each parameter
            if p.is_leaf and p.requires_grad:    # only leaf params that actually need gradients
                p.register_post_accumulate_grad_hook(self._hook)  # attach hook to param

        self._should_all_reduce = True           # flag to enable/disable gradient syncing
        self.handles = []                        # store async all_reduce handles for later waiting
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1  # total ranks

    def _hook(self, p: Tensor) -> None:          # hook called when gradient for param p is ready
        if p.grad is not None and self._should_all_reduce:  # only if gradient exists + syncing enabled
            # async SUM -------------------------------------------------------
            # async_op=True => starts all_reduce but DOES NOT block (returns handle)
            h = dist.all_reduce(p.grad, dist.ReduceOp.SUM, async_op=True)  # launch SUM
            self.handles.append((h, p))          # remember handle + which param it belongs to
    #not used in this train calling 
    @contextmanager
    def no_sync(self):                           # context manager to skip all_reduce (like real DDP.no_sync)
        before = self._should_all_reduce         # save current flag
        self._should_all_reduce = False          # disable syncing inside the block
        try:
            yield                                # run user's code
        finally:
            self._should_all_reduce = before     # restore old flag when leaving the block

    def forward(self, *args, **kwargs):          # forward just calls underlying module
        return self.module(*args, **kwargs)      # normal forward pass
    #addtional check and for division(average gradient)
    def finish_gradient_synchronization(self) -> float:  # wait for async comm + average grads
        t0 = time.monotonic()                    # start timing wait + averaging

        # wait all handles ----------------------------------------------------
        # ensures every async all_reduce is finished before optimizer step
        # print(self.handles)
        for h, _ in self.handles:                # loop over each async handle
            print(h)
            h.wait()                             # block until that all_reduce completes

        # IMPORTANT: avg after SUM -------------------------------------------
        # after all_reduce SUM, grads are summed across ranks
        # now divide to get averaged gradient (DDP behaviour)
        for _, p in self.handles:                # loop again (need param tensors)
            if p.grad is not None:               # safety check
                p.grad.div_(self.world_size)     # in-place divide => average

        self.handles.clear()                     # reset handle list for next iteration
        return time.monotonic() - t0             # return time spent waiting/averaging


class MLP(nn.Module):                            # a simple test network
    def __init__(self):
        super().__init__()                       # base init
        self.net = nn.Sequential(                # define layers
            nn.Linear(10, 64),                   # 10 -> 64
            nn.ReLU(),                           # activation
            nn.Linear(64, 1),                    # 64 -> 1
        )

    def forward(self, x):                        # forward pass
        return self.net(x)                       # apply sequential net


def setup(rank, world_size):                     # init process group for a rank
    os.environ["MASTER_ADDR"] = "127.0.0.1"      # rendezvous address (local machine)
    os.environ["MASTER_PORT"] = "29501"          # rendezvous port (unique port)
    dist.init_process_group(                     # start distributed backend
        "gloo",                                  # CPU backend
        rank=rank,                               # rank id for this process
        world_size=world_size                    # total ranks
    )


def cleanup():                                   # cleanup dist resources
    dist.destroy_process_group()                 # destroy the process group


def train(rank, world_size):                     # training code that runs in each process
    setup(rank, world_size)                      # init dist for this rank
    torch.manual_seed(0)                         # set seed

    model = DDP(MLP())                           # wrap model in our custom DDP
    opt = optim.SGD(model.parameters(), lr=0.1)  # optimizer on model params
    loss_fn = nn.MSELoss()                       # regression loss

    for step in range(5):                        # run some training steps
        x = torch.randn(4, 10)                   # random input batch (per rank)
        y = torch.randn(4, 1)                    # random target batch (per rank)

        opt.zero_grad()                          # clear old grads
        pred = model(x)                          # forward pass
        loss = loss_fn(pred, y)                  # compute loss
        loss.backward()                          # backward: hooks launch async all_reduce per param

        # at this moment, all_reduce ops may still be running in background
        comm_time = model.finish_gradient_synchronization()  # wait for all + average grads

        opt.step()                               # update weights (now grads are synced/averaged)

        if rank == 0:                            # only rank0 prints
            print(
                f"[ASYNC-HOOKS] step={step} loss={loss.item():.4f} wait={comm_time*1000:.2f}ms"
            )

    cleanup()                                    # shutdown dist backend


if __name__ == "__main__":                       # script entry point
    world_size = 2                               # number of ranks/processes
    mp.spawn(                                    # spawn world_size processes
        train,                                   # function to run in each process
        args=(world_size,),                      # extra args to train()
        nprocs=world_size,                       # create exactly world_size processes
        join=True                                # wait for all processes to finish
    )
