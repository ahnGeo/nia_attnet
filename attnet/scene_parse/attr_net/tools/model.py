import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models


PYTORCH_VER = torch.__version__


class AttributeNetwork(nn.Module):

    def __init__(self, opt):    
        super(AttributeNetwork, self).__init__()
        if opt.concat_img:
            self.input_channels = 6
        else:
            self.input_channels = 3

        if opt.load_checkpoint_path:
            print('| loading checkpoint from %s' % opt.load_checkpoint_path)
            checkpoint = torch.load(opt.load_checkpoint_path)
            if self.input_channels != checkpoint['input_channels']:
                raise ValueError('Incorrect input channels for loaded model')
            self.output_dim = checkpoint['output_dim']
            self.net = _Net(self.output_dim, self.input_channels).to("cuda")
            self.net.load_state_dict(checkpoint['model_state'])
        else:
            print('| creating new model')
            
            if "clevr" in opt.dataset:
                output_dims = {
                    'clevr': 18
                }
                self.output_dim = output_dims[opt.dataset]
                
            #* for ours
            if "basketball" in opt.dataset:
                self.output_dim = opt.feature_vector_len
                print("| output dimension is %d" % opt.feature_vector_len) 
            
            self.net = _Net(self.output_dim, self.input_channels).to("cuda")

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.learning_rate)

        # self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        # self.gpu_ids = opt.gpu_ids
        # if self.use_cuda:
        #     self.net.cuda(opt.gpu_ids[0])       
        self.use_cuda=True
        self.input, self.label = None, None
                
    def set_input(self, x, y=None):
        self.input = self._to_var(x)
        if y is not None:
            self.label = self._to_var(y)

    def step(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss.backward()
        self.optimizer.step()

    def forward(self):
        self.pred = self.net(self.input)
        if self.label is not None:
            self.loss = self.criterion(self.pred, self.label)
            
    def get_loss(self):
        if PYTORCH_VER.startswith('0.4'):
            return self.loss.data.item()
        else:
            return self.loss.data

    def get_pred(self):
        return self.pred.data.cpu().numpy()
    
    #*####
    def check_correct(self, att_len_list) :
        #* (ex) att_len_list = [0, 2, 5, 3]
        #* pred.shape (B, output_dim)
        
        score = torch.zeros((self.pred.shape[0], len(att_len_list[1:])))  #* initialize score list
        for i in range(self.pred.shape[0]) :
            start_idx = 0
            end_idx = 0
            for j in range(len(att_len_list[1:])) : 
                start_idx += att_len_list[j]
                end_idx += att_len_list[j + 1]
                if torch.argmax(self.pred[i][start_idx : end_idx]) == torch.argmax(self.label[i][start_idx : end_idx]) :  
                    score[i, j] += 1
                    
        score = torch.sum(score, axis=0)  #* score.shape = (1, num of atts)
        
        return score
    #*####

    def eval_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def save_checkpoint(self, save_path,opt):
        checkpoint = {
            'input_channels': self.input_channels,
            'output_dim': self.output_dim,
            'model_state': self.net.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if self.use_cuda:
            self.net.cuda(opt.gpu)

    def _to_var(self, x):
        if self.use_cuda:
            x = x.cuda()    
        return Variable(x)


class _Net(nn.Module):
    def __init__(self, output_dim, input_channels=6):
        super(_Net, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        
        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        output = self.final_layer(x)
        return output


def get_model(opt):
    model = AttributeNetwork(opt)
    return model