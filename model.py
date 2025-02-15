
from lib import *
from video_swin import SwinTransformer3D

class EncImg(T.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.swin = SwinTransformer3D()
        self.swin.load_state_dict(T.load('./_snapshot/ckpt_video-swin.pt', map_location='cpu'))
        
        self.emb_cls = T.nn.Parameter(0.02*T.randn(1, 1, 1, 768))
        self.emb_pos = T.nn.Parameter(0.02*T.randn(1, 1, 1+14**2, 768))
        self.emb_len = T.nn.Parameter(0.02*T.randn(1, 10, 1, 768))
        self.norm = T.nn.LayerNorm(768)
    
    def forward(self, img, img_cls):
        _B, _T, _C, _H, _W = img.shape
        _h, _w = _H//32, _W//32
        
        img = TV.transforms.Normalize([0.485, 0.456, 0.406], 
                                      [0.229, 0.224, 0.225])(img)
        
        f_img = self.swin(img.transpose(1, 2)).transpose(1, 2)
        
        f_img = f_img.permute(0, 1, 3, 4, 2).view([_B, _T, _h*_w, 768])
        if img_cls is not None:
          f_img = T.cat([img_cls.permute(0,2,1,3), f_img], dim=2)
        else:
            f_img = T.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_img], dim=2)
        f_img += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]+self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]
        f_img = self.norm(f_img).view([_B, _T*(1+_h*_w), -1])
        
        m_img = T.ones(1+_h*_w).long().cuda().unsqueeze(0).unsqueeze(0)
        m_img = m_img.expand([_B, _T, -1]).contiguous().view([_B, _T*(1+_h*_w)])
        
        return f_img, m_img

class EncTxt(T.nn.Module):
    def __init__(self):
        super().__init__()
        
        bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.emb_txt = bert.embeddings
    
    def forward(self, txt, txt_cls):
        f_txt = self.emb_txt(txt)
        if txt_cls is not None:
            f_txt = T.cat([txt_cls.squeeze(1), f_txt], dim=1)
        return f_txt

class VIOLET_Base(T.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc_img, self.enc_txt = EncImg(), EncTxt()
        bert = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mask_ext, self.trsfr = bert.get_extended_attention_mask, bert.bert.encoder
    
    def go_feat(self, img, txt, mask, txt_cls, img_cls):
        txt_cls = txt_cls if self.args.use_clip_txt_cls else None
        mask = T.cat([T.ones((mask.shape[0],1)).cuda(), mask], dim=1) if self.args.use_clip_txt_cls else mask
        # img_cls = img_cls if self.args.aggregation == 'vision_CLS' else None
        if self.freeze_vision:
            with T.no_grad():
                feat_img, mask_img = self.enc_img(img, img_cls)
        else:
            feat_img, mask_img = self.enc_img(img, img_cls)

        if self.freeze_lang:
            with T.no_grad():
                feat_txt, mask_txt = self.enc_txt(txt, txt_cls), mask
        else:
            feat_txt, mask_txt = self.enc_txt(txt, txt_cls), mask

        return feat_img, mask_img, feat_txt, mask_txt
    
    def go_cross(self, feat_img, mask_img, feat_txt, mask_txt):
        feat, mask = T.cat([feat_img, feat_txt], dim=1), T.cat([mask_img, mask_txt], dim=1)
        mask = self.mask_ext(mask, mask.shape, mask.device)
        if self.freeze_cross:
            with T.no_grad():
                out = self.trsfr(feat, mask, output_attentions=True)
        else:
            out = self.trsfr(feat, mask, output_attentions=True)
        return out['last_hidden_state'], out['attentions']
    
    def load_ckpt(self, ckpt):
        if ckpt=='':
            print('===== Init VIOLET =====')
            return
        
        ckpt_new, ckpt_old = T.load(ckpt, map_location='cpu'), self.state_dict()
        key_old = set(ckpt_old.keys())
        for k in ckpt_new:
            if k in ckpt_old and ckpt_new[k].shape==ckpt_old[k].shape:
                ckpt_old[k] = ckpt_new[k]
                key_old.remove(k)     
        self.load_state_dict(ckpt_old)
        print('===== Not Load:', key_old, '=====')
        
