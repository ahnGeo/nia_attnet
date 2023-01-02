import os
import json
import torch
import utils as utils

class Trainer:

    def __init__(self, opt, model, train_loader, val_loader=None):
        self.num_iters = opt.num_iters
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.opt=opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.val_epochs = opt.val_epochs
        self.output_dim = opt.feature_vector_len

        self.stats = {
            'train_losses': [],
            'train_losses_ts': [],
            'train_acc' : [],
            'val_losses': [],
            'val_losses_ts': [],
            'val_acc' : [],
            'best_val_loss': 9999,
            'model_t': 0
        }

    def train(self):
        print('| start training, running in directory %s' % self.run_dir)
        t = 0
        epoch = 0
        
        att_len_list = [0, 7, 2, 4, 8, 7, 8, 2, 3, 2]  #@ if use "선수상황"(len 2), "선수방향"(len 4) -> [0, 2, 4], make_basketball_obj_json.py att_bind에서 앞에 위치한 att의 len 먼저 적어야 함
        
        while t < self.num_iters:
            epoch += 1
            
            # score = torch.zeros((1, len(att_len_list[1:]))) 
            # total = 0
            
            for data, label in self.train_loader:    #* data.shape (B=6, 6, 1080, 1920) -> (270, 480)

                t += 1
                self.model.set_input(data, label)
                self.model.step()

                #*####
                # score = torch.add(score, self.model.check_correct(att_len_list))
                # total += data.shape[0]
                #*####

                # if t % self.display_every == 0:
                #     self.stats['train_losses'].append(loss)
                #     print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                #     self.stats['train_losses_ts'].append(t)

                if t % self.checkpoint_every == 0 or t >= self.num_iters:
                    print('| saving checkpoint')
                    self.model.save_checkpoint('%s/checkpoint_iter%08d.pt' %
                                                (self.run_dir, t),self.opt)
                    self.model.save_checkpoint(os.path.join(self.run_dir, 'checkpoint.pt'),self.opt)
                    with open('%s/stats.json' % self.run_dir, 'w') as fout:
                        json_errors = {k: str(v) for k, v in self.stats.items()}
                        json.dump(json_errors, fout)

                if t >= self.num_iters:
                    break
                
            #*## get train loss and acc for every epoch
            loss = self.model.get_loss()
            self.stats['train_losses'].append(loss)
            print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
            # score = torch.flatten(score)
            # acc = [str(float(score[i] / total))[:8] for i in range(score.shape[0])]
            # print("| train acc : {}".format(", ".join(acc)))
            # self.stats['train_acc'].append(", ".join(acc))
            # self.stats['train_losses_ts'].append(t)
            
            #*## validation epoch
            if epoch % self.val_epochs == 0 :
                if self.val_loader is not None:    
                    print('| checking validation loss')
                    val_loss = self.check_val_loss(att_len_list)
                    print('| validation loss %f' % val_loss)
                    if val_loss <= self.stats['best_val_loss']:
                        print('| best model')
                        self.stats['best_val_loss'] = val_loss
                        self.stats['model_t'] = t
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir,self.opt)
                    self.stats['val_losses'].append(val_loss)
                    self.stats['val_losses_ts'].append(t)

    def check_val_loss(self, att_len_list):
        self.model.eval_mode()
        loss = 0
        t = 0
        
        score = torch.zeros((1, len(att_len_list[1:])))
        total = 0
        for x, y in self.val_loader:
            self.model.set_input(x, y)
            self.model.forward()
            loss += self.model.get_loss()
            #*####
            # print(self.model.check_correct(1), self.model.get_pred().shape[0])
            score = torch.add(score, self.model.check_correct(att_len_list))
            total += self.model.get_pred().shape[0]
            #*####
            t += 1
        self.model.train_mode()
        
        #*####
        score = torch.flatten(score)
        acc = [str(float(score[i] / total))[:8] for i in range(score.shape[0])]
        print('| validation acc : {}'.format(", ".join(acc)))   
        self.stats['val_acc'].append(", ".join(acc))                                                                                                                                                     
        #*####
        
        return loss / t if t != 0 else 0


def get_trainer(opt, model, train_loader, val_loader=None):
    return Trainer(opt, model, train_loader, val_loader)