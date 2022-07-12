
from requests import head
from lib import *

class Agent_Base:
    def __init__(self, args, model):
        super().__init__()
        
        self.args, self.model = args, model
        modules = self.model.module
        base_params = list(modules.enc_img.parameters()) + list(modules.enc_txt.parameters()) + list(modules.trsfr.parameters())
        head_params = list(modules.fc.parameters())
        
        self.loss_func = T.nn.CrossEntropyLoss().cuda()
        self.optzr = T.optim.AdamW([{'params': base_params, 'lr': args.lr_base, 'weight_decay': args.decay_base}, {'params': head_params}],
                                    lr=args.lr_head, betas=(0.9, 0.98), weight_decay=args.decay_head)
        self.scaler = T.cuda.amp.GradScaler()
        