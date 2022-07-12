
from lib import *

class Dataset_Base(T.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.tokzr = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    def transfom_img(self, img):
        img = img.convert('RGB')
        w, h = img.size
        img = TV.transforms.Compose([TV.transforms.Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                                     TV.transforms.Resize([self.args.img_size, self.args.img_size]), 
                                     TV.transforms.ToTensor()])(img)
        return img
    
    def str2txt(self, s):
        txt = self.tokzr.encode(s, padding='max_length', max_length=self.args.max_length, truncation=True)
        mask = [1 if w!=0 else w for w in txt]
        txt, mask = np.array(txt, dtype=np.int64), np.array(mask, dtype=np.int64)
        return txt, mask
        