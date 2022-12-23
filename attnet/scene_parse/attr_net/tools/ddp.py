import torch
import os



def init_distributed_mode(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ["RANK"])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        opt.distributed = False
        return

    opt.distributed = True

    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        opt.rank, opt.dist_url, opt.gpu), flush=True)
    torch.distributed.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                         world_size=opt.world_size, rank=opt.rank)
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(opt.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*opt, **kwopt):
        force = kwopt.pop('force', False)
        if is_master or force:
            builtin_print(*opt, **kwopt)

    __builtin__.print = print