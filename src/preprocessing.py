import cv2, numpy as np
from common import load_config

def load_image(path):
    img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_gray(img): return img if img.ndim==2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def binarize(img): return cv2.threshold(to_gray(img),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
def deskew(b):
    pts=np.column_stack(np.where(b>0))
    if pts.shape[0] < 25: return b
    angle=cv2.minAreaRect(pts[:, ::-1].astype(np.float32))[-1]
    angle=90+angle if angle < -45 else angle
    if abs(angle) < 0.25: return b
    h,w=b.shape; M=cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(b, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
def crop_to_content(b):
    ys,xs=np.where(b>0)
    return [] if len(xs)==0 else [b[ys.min():ys.max()+1, xs.min():xs.max()+1]]
def resize(b,size=(224,224)):
    th,tw=size; h,w=b.shape[:2]
    if h==0 or w==0: return np.zeros((th,tw),dtype=np.uint8)
    s=min(tw/w, th/h); nw=max(1,int(round(w*s))); nh=max(1,int(round(h*s)))
    out=np.zeros((th,tw),dtype=np.uint8); r=cv2.resize(b,(nw,nh),interpolation=cv2.INTER_AREA)
    yo=(th-nh)//2; xo=(tw-nw)//2; out[yo:yo+nh, xo:xo+nw]=r; return out
def normalize(rgb, mean, std):
    x=rgb.astype(np.float32)/255.0; x=(x-np.array(mean,dtype=np.float32))/np.array(std,dtype=np.float32)
    return x.transpose(2,0,1).astype(np.float32)
def preprocess(path, config=None):
    config=config or load_config("config.yaml")
    b=resize((crop_to_content(deskew(binarize(load_image(path)))) or [binarize(load_image(path))])[0], tuple(config["preprocessing"]["target_size"]))
    rgb=np.stack([b,b,b],axis=-1)
    return normalize(rgb, config["preprocessing"]["imagenet_mean"], config["preprocessing"]["imagenet_std"])
