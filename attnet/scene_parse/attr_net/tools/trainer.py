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

        self.stats = {
            'train_losses': [],
            'train_losses_ts': [],
            'val_losses': [],
            'val_losses_ts': [],
            'best_val_loss': 9999,
            'model_t': 0
        }

    def train(self):
        print('| start training, running in directory %s' % self.run_dir)
        t = 0
        epoch = 0
        while t < self.num_iters:
            epoch += 1
            for data, label in self.train_loader:
                t += 1
                self.model.set_input(data, label)
                self.model.step()
                loss = self.model.get_loss()

                if t % self.display_every == 0:
                    self.stats['train_losses'].append(loss)
                    print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                    self.stats['train_losses_ts'].append(t)

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
            
            #* validation epoch
            if epoch % self.val_epochs == 0 :
                if self.val_loader is not None:    
                    print('| checking validation loss')
                    val_loss = self.check_val_loss()
                    print('| validation loss %f' % val_loss)
                    if val_loss <= self.stats['best_val_loss']:
                        print('| best model')
                        self.stats['best_val_loss'] = val_loss
                        self.stats['model_t'] = t
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir,self.opt)
                    self.stats['val_losses'].append(val_loss)
                    self.stats['val_losses_ts'].append(t)

    def check_val_loss(self):
        self.model.eval_mode()
        loss = 0
        t = 0
        correct = 0
        total = 0
        for x, y in self.val_loader:
            self.model.set_input(x, y)
            self.model.forward()
            loss += self.model.get_loss()
            #*####
            # print(self.model.check_correct(1), self.model.get_pred().shape[0])
            correct += self.model.check_correct(1)
            total += self.model.get_pred().shape[0]
            #*####
            t += 1
        self.model.train_mode()
        
        #*####
        print('| validation acc about 선수진영 is %f' % (correct / total))
        #*####
        
        return loss / t if t != 0 else 0


def get_trainer(opt, model, train_loader, val_loader=None):
    return Trainer(opt, model, train_loader, val_loader)