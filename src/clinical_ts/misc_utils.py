from pytorch_lightning.callbacks import Callback
import argparse

class LRMonitorCallback(Callback):
    def __init__(self,interval="epoch",start=True,end=True):
        self.interval = interval
        self.start = start
        self.end = end
        
    def on_train_batch_start(self, trainer, *args, **kwargs):                
        if(self.interval == "step" and self.start):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

    def on_train_epoch_start(self, trainer, *args, **kwargs):                
        if(self.interval == "epoch" and self.start):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)
    
    def on_train_batch_end(self, trainer, *args, **kwargs):                
        if(self.interval == "step" and self.end):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

    def on_train_epoch_end(self, trainer, *args, **kwargs):                
        if(self.interval == "epoch" and self.end):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

############################################################################################################
def _freeze_bn_stats(model, freeze=True):
    for m in model.modules():
        if(isinstance(m,nn.BatchNorm1d)):
            if(freeze):
                m.eval()
            else:
                m.train()

############################################################################################################
def sanity_check(model, state_dict_pre):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'head.1.weight' in k or 'head.1.bias' in k:
            continue


        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

############################################################################################################
#from https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/vqvae.py
# -----------------------------------------------------------------------------
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
class DecayTemperature(Callback):
    def __init__(self,num_steps=15000,tstart=1,tend=1.0/16.):
        super(DecayTemperature, self).__init__()
        self.num_steps = num_steps
        self.tstart = tstart
        self.tend = tend
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = cos_anneal(0, self.num_steps, self.tstart, self.tend, trainer.global_step)
        pl_module.model_cpc.quantizer.temperature = t

class RampBeta(Callback):
    def __init__(self,num_steps=5000,betaend=5e-4):
        super(RampBeta, self).__init__()
        self.num_steps = num_steps
        self.betaend = betaend

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, self.num_steps, 0.0, self.betaend, trainer.global_step)
        pl_module.model_cpc.quantizer.kld_scale = t

class DecayLR(Callback):
    def __init__(self,num_steps=1200000,lrstart=3e-4,lrend=1.25e-6):
        super(DecayLR, self).__init__()
        self.num_steps = num_steps
        self.lrstart = lrstart
        self.lrend = lrend

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, self.num_steps, self.lrstart, self.lrend, trainer.global_step)
        for g in pl_module.model_cpc.optimizer.param_groups:
            g['lr'] = t
            
           
#############################################################################################################
# DEFAULT ARGPARSE


def add_default_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
    parser.add_argument('--data', metavar='DIR',type=str,
                        help='path(s) to dataset (comma-separated)')#,action='append')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default='adam', help='sgd/adam')#was sgd
    parser.add_argument('--output-path', default='.', type=str,dest="output_path",
                        help='output path')
    parser.add_argument('--metadata', default='', type=str,
                        help='metadata for output')
    
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=1, help="number of compute nodes")
    parser.add_argument("--precision", type=int, default=16, help="16/32")
    parser.add_argument("--distributed-backend", dest="distributed_backend", type=str, default=None, help="None/ddp")
    parser.add_argument("--accumulate", type=int, default=1, help="accumulate grad batches (total-bs=accumulate-batches*bs)")
        
    parser.add_argument("--input-size", dest="input_size", type=int, default=16000)
    parser.add_argument("--train-head-only", action="store_true", help="freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    parser.add_argument("--finetune", action="store_true", help="finetuning (downstream classification task)",  default=False )
    parser.add_argument("--linear-eval", action="store_true", help="linear evaluation instead of full finetuning",  default=False )
    
    parser.add_argument("--lr-schedule", type=str, help="const/warmup-const/warmup-cos/warmup-cos-restart/warmup-poly", default="const")
    parser.add_argument("--lr-num-warmup-steps", type=int, help="number of linear lr warmup steps", default=1000)
    
    parser.add_argument("--discriminative-lr-factor", type=float, help="factor by which the lr decreases per layer group during finetuning", default=0.1)
    parser.add_argument("--lr-find", action="store_true", help="run lr finder before training run", default=False)
    parser.add_argument("--auto-batch-size", action='store_true')

    parser.add_argument("--auc-maximization", action="store_true", help="direct auc maximization",  default=False)
    parser.add_argument("--refresh-rate", type=int, help="progress bar refresh rate (0 to disable)", default=0)

    parser.add_argument("--mlflow", action="store_true", help="also log to mlflow")
    
    return parser
