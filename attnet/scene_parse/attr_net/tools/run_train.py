import sys
from options import get_options
from datasets import get_dataloader,get_dataset
from model import get_model
from trainer import get_trainer
from ddp import init_distributed_mode
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch

#!###################
torch.manual_seed(777)

def main(opt):
    
    init_distributed_mode(opt)
    train_datasets = get_dataset(opt, 'train')
    val_datasets = get_dataset(opt, 'val')
    
    cudnn.benchmark = True
    global_rank=dist.get_rank()
    num_tasks=dist.get_world_size()
    sampler_train=torch.utils.data.DistributedSampler(
        train_datasets,num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        val_datasets, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    
    
    data_loader_train = torch.utils.data.DataLoader(
        train_datasets, sampler=sampler_train,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    data_loader_val = torch.utils.data.DataLoader(
            val_datasets, sampler=sampler_val,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False
        )
    total_batch_size = opt.batch_size  * dist.get_world_size()
    print(f"total_batch_size is {total_batch_size}")
    model = get_model(opt)
    device = torch.device(opt.gpu)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    model_without_ddp = model.module
    trainer = get_trainer(opt, model_without_ddp, data_loader_train, data_loader_val)
    trainer.train() 

if __name__ == "__main__":
    opt = get_options('train')
    main(opt)