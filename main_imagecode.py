
from pathlib import Path
from lib import *
from dataset import Dataset_Base
from model import VIOLET_Base
from agent import Agent_Base
import wandb
import argparse

def load_data(train=True):
    def format_data(json_file):
        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path('/network/scratch/b/benno.krojer/dataset/games') / img_dir).glob("*.jpg"))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                dataset.append((img_dir, img_files, int(img_idx), text))
        return dataset
    if train:
        data = json.load(open('../imagecode/data/train_data.json', 'r'))
    else:
        data = json.load(open('../imagecode/data/valid_data.json', 'r'))
    data = format_data(data)
    return data

class Dataset_Retrieval(Dataset_Base):
    def __init__(self, args, split):
        super().__init__(args)
        self.data = load_data(split=='train')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.data[idx]
        
        images = [self.transfom_img(Image.open(photo_file)) for photo_file in img_files]
        img = T.cat(images, dim=0)
        
        self.txt, mask = self.str2txt(text)
        
        return img, self.txt, img_idx, mask

class VIOLET_Retrieval(VIOLET_Base):
    def __init__(self):
        super().__init__()
        
        self.avgpool = T.nn.AdaptiveAvgPool1d(1)

        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), 
                                    T.nn.Linear(768, 768*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(768*2, 1)])
    
    def forward(self, img, txt, mask):
        img = img.reshape((img.shape[0], 10, 3, img.shape[-2], img.shape[-1]))  
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32
        
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask) #B,10,3,H,W -> B,10,250,768
        
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out = out.reshape(_B, 12, (1+_h*_w), -1)
        out = self.fc(out[:,:10,0,:]).squeeze()
        
        return out

class Agent_Retrieval(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
    
    def step(self, img, txt, img_idx, mask, is_train):
        self.optzr.zero_grad()
        with T.cuda.amp.autocast():
            out = self.model(img.cuda(), txt.cuda(), mask.cuda())
            ls = self.loss_func(out, img_idx.cuda())
        if is_train==True:
            self.scaler.scale(ls).backward()
            self.scaler.step(self.optzr)
            self.scaler.update()
            return ls.item()
        else:
            out = T.argmax(out, dim=1)
            ac = (out==img_idx.cuda()).float().mean().item()
            return ac
    
    def go_dl(self, dl, is_train):
        ret = []
        for img, txt, img_idx, mask in tqdm(dl, ascii=True):
            ret.append(self.step(img, txt, img_idx, mask, is_train))
        ret = float(np.average(ret))
        
        return ret
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_head', type=float)
    parser.add_argument('--lr_base', type=float)
    parser.add_argument('--decay_head', type=float)
    parser.add_argument('--decay_base', type=float)
    parser.add_argument('--lr_head', type=float)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--freeze_lang', type=bool)
    parser.add_argument('--freeze_vision', type=bool)
    parser.add_argument('--freeze_cross', type=bool)
    parser.add_argument('--video_only', type=bool)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--activation', type=str)

    args = parser.parse_args()
    wandb.init(project='violet', settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)

    args = json.load(open('_data/args_msrvtt-retrieval.json', 'r'))
    args['size_batch'] = args['size_batch']*T.cuda.device_count()
    args['path_output'] = '_snapshot/_%s_%s'%(args['task'], datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(args['path_output'], exist_ok=True)
    json.dump(args, open('%s/args.json'%(args['path_output']), 'w'), indent=2)
    print(args)
    
    dl_tr, dl_vl = [T.utils.data.DataLoader(Dataset_Retrieval(args, split), 
                                                   batch_size=args['size_batch'], shuffle=(split=='train'), 
                                                   num_workers=32, pin_memory=True)\
                           for split in ['train', 'val']]
    
    log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}
    json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
    
    model = T.nn.DataParallel(VIOLET_Retrieval().cuda())
    model.module.load_ckpt(args['path_ckpt'])
    if None:
        T.save(model.module.state_dict(), '%s/ckpt_violet_%s_0.pt'%(args['path_output'], args['task']))
    
    agent = Agent_Retrieval(args, model)
    for e in tqdm(range(args['size_epoch']), ascii=True):
        model.train()
        ls_tr = agent.go_dl(dl_tr, True)
        
        model.eval()
        ac_vl = agent.go_dl(dl_vl, False)
        
        log['ls_tr'].append(ls_tr), log['ac_vl'].append(ac_vl)
        json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
        T.save(model.module.state_dict(), '%s/ckpt_violet_%s_%d.pt'%(args['path_output'], args['task'], e+1))
        print('Ep %d: %.6f %.6f'%(e+1, ls_tr, ac_vl))
        