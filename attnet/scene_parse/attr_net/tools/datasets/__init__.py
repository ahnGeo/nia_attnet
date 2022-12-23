from torch.utils.data import DataLoader
from .clevr_object import ClevrObjectDataset
from .basketball_object import BasketballDataset, BasketballTestDataset

def get_dataset(opt, split):
    if opt.dataset == 'clevr':
        if split == 'train':
            ds = ClevrObjectDataset(opt.clevr_mini_ann_path, opt.clevr_mini_img_dir, 'mini',
                                    max_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'val':
            ds = ClevrObjectDataset(opt.clevr_mini_ann_path, opt.clevr_mini_img_dir, 'mini',
                                    min_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'test':
            ds = ClevrObjectDataset(
                opt.clevr_val_ann_path, opt.clevr_val_img_dir, 'val', concat_img=opt.concat_img)
        else:
            raise ValueError('Invalid dataset split11: %s' % split)
    elif opt.dataset == 'basketball':
        if split == 'train':
            ds = BasketballDataset(opt.basketball_ann_path, opt.basketball_img_dir,split='train',
                                    max_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'val':
            ds = BasketballDataset(opt.basketball_ann_path, opt.basketball_img_dir,split='val',
                                    min_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'test':
            ds = BasketballTestDataset(opt.basketball_test_ann_path, opt.basketball_test_img_dir,split='val',
                                    min_img_id=opt.split_id, concat_img=opt.concat_img)
        else:
            raise ValueError('Invalid dataset split: %s' % split)
    elif opt.dataset == 'basketball_mini':
        if split == 'train':
            ds = BasketballDataset(opt.basketball_ann_path, opt.basketball_img_dir,split='train',
                                    max_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'val':
            ds = BasketballDataset(opt.basketball_ann_path, opt.basketball_img_dir,split='val',
                                    min_img_id=opt.split_id, concat_img=opt.concat_img)
        elif split == 'test':
            ds = BasketballTestDataset(opt.basketball_test_ann_path, opt.basketball_test_img_dir,split='val',
                                    min_img_id=opt.split_id, concat_img=opt.concat_img)
        else:
            raise ValueError('Invalid dataset split: %s' % split)
    return ds   


def get_dataloader(opt, split):
    ds = get_dataset(opt, split)
    loader = DataLoader(dataset=ds, batch_size=opt.batch_size,
                        num_workers=opt.num_workers, shuffle=opt.shuffle_data)
    return loader