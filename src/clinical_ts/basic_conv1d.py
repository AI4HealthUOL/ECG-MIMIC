__all__ = ['AdaptiveConcatPool1d', 'SqueezeExcite1d',
           'weight_init', 'create_head1d', 'basic_conv1d', 'fcn', 'fcn_wang', 'schirrmeister', 'sen', 'basic1d']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Iterable

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False): 
        super().__init__()
        self.full = full
    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)



def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None, layer_norm=False, permute=False):
    '''
    Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`. 
    permute for input of the form B,Seq,Feat"
    '''
    layers=[]
    if(permute):
        layers.append(LambdaLayer(lambda x: x.permute(0,2,1)))
    if(bn):
        if(layer_norm is False):
            layers.append(nn.BatchNorm1d(n_in))
        else:
            layers.append(nn.LayerNorm(n_in))
    if(permute):
        layers.append(LambdaLayer(lambda x: x.permute(0,2,1)))
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

# Cell

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

def _conv1d(in_planes,out_planes,kernel_size=3, stride=1, dilation=1, act="relu", bn=True, drop_p=0, layer_norm=False):
    lst=[]
    if(drop_p>0):
        lst.append(nn.Dropout(drop_p))
    lst.append(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation, bias=not(bn)))
    if(bn):
        if(layer_norm):
            lst.append(LambdaLayer(lambda x: x.transpose(1,2)))
            lst.append(nn.LayerNorm(out_planes))
            lst.append(LambdaLayer(lambda x: x.transpose(1,2)))
        else:
            lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    if(act=="gelu"):
        lst.append(nn.GELU())
    return nn.Sequential(*lst)

def _fc(in_planes,out_planes, act="relu", bn=True):
    lst = [nn.Linear(in_planes, out_planes, bias=not(bn))]
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

# Cell
class SqueezeExcite1d(nn.Module):
    '''squeeze excite block as used for example in LSTM FCN'''
    def __init__(self,channels,reduction=16):
        super().__init__()
        channels_reduced = channels//reduction
        self.w1 = torch.nn.Parameter(torch.randn(channels_reduced,channels).unsqueeze(0))
        self.w2 = torch.nn.Parameter(torch.randn(channels, channels_reduced).unsqueeze(0))

    def forward(self, x):
        #input is bs,ch,seq
        z=torch.mean(x,dim=2,keepdim=True)#bs,ch
        intermed = F.relu(torch.matmul(self.w1,z))#(1,ch_red,ch * bs,ch,1) = (bs, ch_red, 1)
        s=F.sigmoid(torch.matmul(self.w2,intermed))#(1,ch,ch_red * bs, ch_red, 1=bs, ch, 1
        return s*x #bs,ch,seq * bs, ch,1 = bs,ch,seq

# Cell
def weight_init(m):
    '''call weight initialization for model n via n.appy(weight_init)'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    if isinstance(m,SqueezeExcite1d):
        stdv1=math.sqrt(2./m.w1.size[0])
        nn.init.normal_(m.w1,0.,stdv1)
        stdv2=math.sqrt(1./m.w2.size[1])
        nn.init.normal_(m.w2,0.,stdv2)

# Cell
def create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = [ps] if not isinstance(ps,Iterable) else ps
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.AdaptiveAvgPool1d(1), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    return nn.Sequential(*layers)

# Cell
class basic_conv1d(nn.Sequential):
    '''basic conv1d'''
    def __init__(self, filters=[128,128,128,128],kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,split_first_layer=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        layers = []
        if(isinstance(kernel_size,int)):
            kernel_size = [kernel_size]*len(filters)
        for i in range(len(filters)):
            layers_tmp = []

            layers_tmp.append(_conv1d(input_channels if i==0 else filters[i-1],filters[i],kernel_size=kernel_size[i],stride=(1 if (split_first_layer is True and i==0) else stride),dilation=dilation,act="none" if ((headless is True and i==len(filters)-1) or (split_first_layer is True and i==0)) else act, bn=False if (headless is True and i==len(filters)-1) else bn,drop_p=(0. if i==0 else drop_p)))
            if((split_first_layer is True and i==0)):
                layers_tmp.append(_conv1d(filters[0],filters[0],kernel_size=1,stride=1,act=act, bn=bn,drop_p=0.))
                #layers_tmp.append(nn.Linear(filters[0],filters[0],bias=not(bn)))
                #layers_tmp.append(_fc(filters[0],filters[0],act=act,bn=bn))
            if(pool>0 and i<len(filters)-1):
                layers_tmp.append(nn.MaxPool1d(pool,stride=pool_stride,padding=(pool-1)//2))
            if(squeeze_excite_reduction>0):
                layers_tmp.append(SqueezeExcite1d(filters[i],squeeze_excite_reduction))
            layers.append(nn.Sequential(*layers_tmp))

        #head
        #layers.append(nn.AdaptiveAvgPool1d(1))
        #layers.append(nn.Linear(filters[-1],num_classes))
        #head #inplace=True leads to a runtime error see ReLU+ dropout https://discuss.pytorch.org/t/relu-dropout-inplace/13467/5
        self.headless = headless
        if(headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1),Flatten())
        else:
            head=create_head1d(filters[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)

        super().__init__(*layers)

    def get_layer_groups(self):
        return (self[2],self[-1])

    def get_output_layer(self):
        if self.headless is False:
            return self[-1][-1]
        else:
            return None

    def set_output_layer(self,x):
        if self.headless is False:
            self[-1][-1] = x


# Cell
def fcn(filters=[128]*5,num_classes=2,input_channels=8,**kwargs):
    filters_in = filters + [num_classes]
    return basic_conv1d(filters=filters_in,kernel_size=3,stride=1,pool=2,pool_stride=2,input_channels=input_channels,act="relu",bn=True,headless=True)

def fcn_wang(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=[128,256,128],kernel_size=[8,5,3],stride=1,pool=0,pool_stride=2, num_classes=num_classes,input_channels=input_channels,act="relu",bn=True,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def schirrmeister(num_classes=2,input_channels=8,kernel_size=10,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=[25,50,100,200],kernel_size=kernel_size, stride=3, pool=3, pool_stride=1, num_classes=num_classes, input_channels=input_channels, act="relu", bn=True, headless=False,split_first_layer=True,drop_p=0.5,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def sen(filters=[128]*5,num_classes=2,input_channels=8,kernel_size=3,squeeze_excite_reduction=16,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=filters,kernel_size=kernel_size,stride=2,pool=0,pool_stride=0,input_channels=input_channels,act="relu",bn=True,num_classes=num_classes,squeeze_excite_reduction=squeeze_excite_reduction,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def basic1d(filters=[128]*5,kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
    return basic_conv1d(filters=filters,kernel_size=kernel_size, stride=stride, dilation=dilation, pool=pool, pool_stride=pool_stride, squeeze_excite_reduction=squeeze_excite_reduction, num_classes=num_classes, input_channels=input_channels, act=act, bn=bn, headless=headless,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)
