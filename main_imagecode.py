
from pathlib import Path
from lib import *
from dataset import Dataset_Base
from model import VIOLET_Base
from agent import Agent_Base
import wandb
import argparse

def load_data(train=True, video_only=False):

    

    def format_data(json_file, cache_file):
        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path('/network/scratch/b/benno.krojer/dataset/games') / img_dir).glob("*.jpg"))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open_images' in img_dir
                txt_cls = cache_file['txt'][f'{img_dir}_[{img_idx}]']
                img_cls = cache_file['img'][img_dir]
                if video_only:
                    if not static:
                        dataset.append((img_dir, img_files, int(img_idx), text, txt_cls, img_cls))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text, txt_cls, img_cls))
        return dataset
    if train:
        data = json.load(open('../imagecode/data/train_data.json', 'r'))
        cached_clip = pickle.load(open('/network/scratch/b/benno.krojer/dataset/clip_cls_train.pkl', 'rb'))
    else:
        data = json.load(open('../imagecode/data/valid_data.json', 'r'))
        cached_clip = pickle.load(open('/network/scratch/b/benno.krojer/dataset/clip_cls_valid.pkl', 'rb'))
    data = format_data(data, cached_clip)
    return data

def load_data_debug(train=True, video_only=False):
    def format_data(json_file):
        dataset = []
        i = 0
        for img_dir, data in json_file.items():
            img_files = list((Path('/network/scratch/b/benno.krojer/dataset/games') / img_dir).glob("*.jpg"))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open_images' in img_dir
                if True:
                    if not static and i < 100:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                        i += 1
                else:
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
        self.data = load_data(split=='train', args.video_only)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text, txt_cls, img_cls = self.data[idx]
        
        images = [self.transfom_img(Image.open(photo_file)) for photo_file in img_files]
        img = T.cat(images, dim=0)
        
        self.txt, mask = self.str2txt(text)
        
        return img, self.txt, img_idx, mask, txt_cls, img_cls

class VIOLET_Retrieval(VIOLET_Base):
    def __init__(self, args):
        super().__init__()
        self.aggregation = args.aggregation
        if args.activation == 'relu':
            activation = T.nn.ReLU(inplace=True)
        else:
            activation = T.nn.GELU()
        out_shape = 10 if self.aggregation == 'lang_CLS' else 1
        in_shape = 50 if self.aggregation == 'mean_pooling_features' else 768
        self.fc = T.nn.Sequential(*[T.nn.Dropout(args.dropout), 
                                    T.nn.Linear(in_shape, in_shape*2), activation, 
                                    T.nn.Linear(in_shape*2, out_shape)])
        self.freeze_lang = args.freeze_lang
        self.freeze_vision = args.freeze_vision
        self.freeze_cross = args.freeze_cross
        self.avgpool = T.nn.AdaptiveAvgPool1d(1)
        self.args = args
    
    def forward(self, img, txt, mask, txt_cls, img_cls):
        img = img.reshape((img.shape[0], 10, 3, img.shape[-2], img.shape[-1]))  
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32
        
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask, txt_cls, img_cls) #B,10,3,H,W -> B,10,250,768

        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        if args.aggregation == 'lang_CLS':
            out = self.fc(out[:, (1+_h*_w)*_T, :]) / 0.05
        elif args.aggregation == 'mean_pooling_features':
            out = out[:, :(1+_h*_w)*_T, :]
            out = out.reshape(_B*10, (1+_h*_w), -1)
            out = self.avgpool(out).squeeze()
            out = self.fc(out)
            out = out.reshape(_B, 10)
        elif args.aggregation == 'mean_pooling_patches':
            out = out[:, :(1+_h*_w)*_T, :]
            out = out.reshape(_B*10, (1+_h*_w), -1)
            out = out.permute(0,2,1)
            out = self.avgpool(out).squeeze()
            out = self.fc(out)
            out = out.reshape(_B, 10)
        else:
            out = out.reshape(_B, 12, (1+_h*_w), -1)
            out = self.fc(out[:,:10,0,:]).squeeze()

        return out

class Agent_Retrieval(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.len_dl = 0
    
    def step(self, step, img, txt, img_idx, mask, txt_cls, img_cls, is_train):
        self.optzr.zero_grad()
        with T.cuda.amp.autocast():
            out = self.model(img.cuda(), txt.cuda(), mask.cuda(), txt_cls, img_cls)
            ls = self.loss_func(out, img_idx.cuda())
        if is_train==True:
            self.scaler.scale(ls).backward()
            if step % args.grad_accumulation == 0 or step -1 >= self.len_dl:
                self.scaler.step(self.optzr)
                self.scaler.update()
            return ls.item()
        else:
            out = T.argmax(out, dim=1)
            ac = (out==img_idx.cuda()).float().mean().item()
            return ac
    
    def go_dl(self, dl, is_train):
        self.len_dl = len(dl)
        ret = []
        i = 0
        for img, txt, img_idx, txt_cls, img_cls, mask in tqdm(dl, ascii=True):
            ret.append(self.step(i, img, txt, img_idx, mask, txt_cls, img_cls, is_train))
            i += 1
        ret = float(np.average(ret))
        
        return ret
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_head', type=float)
    parser.add_argument('--lr_base', type=float)
    parser.add_argument('--decay_head', type=float)
    parser.add_argument('--decay_base', type=float)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--freeze_lang', type=str)
    parser.add_argument('--freeze_vision', type=str)
    parser.add_argument('--freeze_cross', type=str)
    parser.add_argument('--video_only', type=str)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--activation', type=str)
    parser.add_argument('--ckpt_path', default="./_data/ckpt_violet_pretrain.pt")
    parser.add_argument('--epochs', default=40)
    parser.add_argument('--aggregation', choices=['vision_CLS', 'mean_pooling_features', 'mean_pooling_patches'])
    parser.add_argument('--use_clip_txt_cls', type=str)
    parser.add_argument('--grad_accumulation', default=1)
    parser.add_argument('--max_length', type=int)
    parser.add_argument("--job_id")

    args = parser.parse_args()
    args.freeze_lang = args.freeze_lang == 'True'
    args.freeze_vision = args.freeze_vision == 'True'
    args.freeze_cross = args.freeze_cross == 'True'
    args.video_only = args.video_only == 'True'
    args.use_clip_txt_cls = args.use_clip_txt_cls == 'True'
    args.grad_accumulation = args.batchsize // 4
    args.batchsize = 4
    
    wandb.init(project='violet', settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)

    args.batchsize = args.batchsize*T.cuda.device_count()
    path_output = '_snapshot/_%s'%(datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(path_output, exist_ok=True)
    print(args)
    
    dl_tr, dl_vl = [T.utils.data.DataLoader(Dataset_Retrieval(args, split), 
                                                   batch_size=args.batchsize, shuffle=(split=='train'), 
                                                   num_workers=32, pin_memory=True, multiprocessing_context='spawn')\
                           for split in ['train', 'val']]
    
    log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}
    json.dump(log, open('%s/log.json'%(path_output), 'w'), indent=2)
    
    model = T.nn.DataParallel(VIOLET_Retrieval(args).cuda())
    model.module.load_ckpt(args.ckpt_path)
    if None:
        T.save(model.module.state_dict(), '%s/ckpt_violet_%s_0.pt'%(path_output, args['task']))
    
    agent = Agent_Retrieval(args, model)
    for e in tqdm(range(args.epochs), ascii=True):
        model.train()
        ls_tr = agent.go_dl(dl_tr, True)
        wandb.log({'Loss': ls_tr})
        model.eval()
        ac_vl = agent.go_dl(dl_vl, False)
        wandb.log({'Validation Accuracy': ac_vl})
        
        log['ls_tr'].append(ls_tr), log['ac_vl'].append(ac_vl)
        json.dump(log, open('%s/log.json'%(path_output), 'w'), indent=2)
        #T.save(model.module.state_dict(), '%s/ckpt_violet_%s.pt'%(path_output, e+1))
        print('Ep %d: %.6f %.6f'%(e+1, ls_tr, ac_vl))
        