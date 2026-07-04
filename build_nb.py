import json
from pathlib import Path
from textwrap import dedent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


cells = []

cells.append(md("""
# Graphology AI Emotion Detection

Colab-ready notebook for an overbuilt handwriting pipeline across both IAM and CVL, expanded to 6 pseudo-emotion classes:
`happy`, `stressed`, `angry`, `calm`, `sad`, `confident`.

The notebook bootstraps a structured backend project on disk, then runs pseudo-labeling, training, inference, and evaluation.
"""))

cells.append(code("""
PROJECT_ROOT = r"C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection"
!pip -q install pyyaml pandas numpy scikit-learn matplotlib seaborn pillow tqdm opencv-python-headless psutil vaderSentiment easyocr
from pathlib import Path
Path(PROJECT_ROOT).mkdir(parents=True, exist_ok=True)
print(PROJECT_ROOT)
"""))

cells.append(code("""
from pathlib import Path
import yaml

root = Path(r"C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection")
root.mkdir(parents=True, exist_ok=True)
for rel in ["src", "data/raw/iam/iam_words/words", "data/raw/cvl", "data/feature_cache", "checkpoints", "logs", "output/evaluation"]:
    (root / rel).mkdir(parents=True, exist_ok=True)

config = {
    "project": {"class_names": ["happy", "stressed", "angry", "calm", "sad", "confident"]},
    "preprocessing": {"imagenet_mean": [0.485, 0.456, 0.406], "imagenet_std": [0.229, 0.224, 0.225], "target_size": [224, 224]},
    "feature_extractor": {"feature_dim": 12, "cache_dir": "data/feature_cache"},
    "label_generator": {"output_csv": "data/labels.csv", "confidence_threshold": 0.55},
    "dataset": {"split_ratios": [0.70, 0.15, 0.15], "random_seed": 42, "confidence_threshold": 0.55},
    "model": {"num_classes": 6, "handcrafted_dim": 12, "projection_dim": 96, "fusion_hidden_dims": [512, 256], "dropout": 0.35, "pretrained": True},
    "training": {"learning_rate": 1e-4, "batch_size": 16, "epochs": 12, "patience": 4, "grad_clip_max_norm": 1.0, "amp": True, "imbalance_strategy": "weighted_loss", "random_seed": 42, "num_workers": 2},
    "paths": {"best_model": "checkpoints/best_model.pt", "scaler_path": "checkpoints/feature_scaler.pkl", "log_dir": "logs", "evaluation_dir": "output/evaluation"},
    "nlp_fusion": {"enabled": False, "fusion_weight": 0.4, "high_arousal_terms": ["hate", "furious", "rage", "panic"]},
    "evaluation": {"calibration": True, "calibration_bins": 10},
}
with open(root / "config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
print("config.yaml written")
"""))

cells.append(md("""
Place IAM images under `C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection\\data\\raw\\iam\\iam_words\\words\\` and CVL images under `C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection\\data\\raw\\cvl\\`.

This notebook is configured for local Windows use in your current project folder.
"""))

cells.append(code(r'''
from pathlib import Path
from textwrap import dedent

root = Path(r"C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection")
files = {}

files["src/common.py"] = dedent("""
import json, os, random, warnings
from pathlib import Path
import numpy as np, torch, yaml

DEFAULT_CONFIG = {"project":{"class_names":["happy","stressed","angry","calm","sad","confident"]},"preprocessing":{"imagenet_mean":[0.485,0.456,0.406],"imagenet_std":[0.229,0.224,0.225],"target_size":[224,224]},"feature_extractor":{"feature_dim":12,"cache_dir":"data/feature_cache"},"dataset":{"split_ratios":[0.70,0.15,0.15],"random_seed":42,"confidence_threshold":0.55}}

def deep_update(a,b):
    x=dict(a)
    for k,v in (b or {}).items():
        x[k]=deep_update(x[k],v) if isinstance(v,dict) and isinstance(x.get(k),dict) else v
    return x
def load_config(path="config.yaml"):
    if not Path(path).exists():
        warnings.warn("config.yaml not found; using defaults")
        return DEFAULT_CONFIG
    return deep_update(DEFAULT_CONFIG, yaml.safe_load(open(path, "r", encoding="utf-8")) or {})
def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True); return Path(path)
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"]=str(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def list_images(root):
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts)
def save_json(obj,path):
    ensure_dir(Path(path).parent)
    json.dump(obj, open(path,"w",encoding="utf-8"), indent=2)
""")

files["src/preprocessing.py"] = dedent("""
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
""")

files["src/feature_label.py"] = dedent("""
import hashlib, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from common import ensure_dir, list_images, load_config
from preprocessing import load_image, binarize, deskew, crop_to_content, resize, to_gray

class FeatureExtractor:
    feature_names=["stroke_thickness_mean","stroke_thickness_std","slant_angle_mean","slant_angle_std","inter_letter_spacing_mean","inter_letter_spacing_std","baseline_deviation","loop_density","char_height_mean","char_height_std","ink_density","connected_component_count"]
    def __init__(self, config=None):
        self.config=config or load_config("config.yaml"); self.feature_dim=12; self.cache_dir=ensure_dir(self.config["feature_extractor"]["cache_dir"])
    def _cache_path(self,path): return self.cache_dir / (hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16] + ".npy")
    def extract(self,path):
        cp=self._cache_path(path)
        if cp.exists():
            arr=np.load(cp)
            if arr.shape[0]==self.feature_dim: return arr.astype(np.float32)
            cp.unlink(missing_ok=True)
        img=load_image(path); b=binarize(img); b=deskew(b); b=resize((crop_to_content(b) or [b])[0], tuple(self.config["preprocessing"]["target_size"]))
        ink=(to_gray(np.stack([b,b,b],axis=-1))>0).astype(np.uint8)
        dist=cv2.distanceTransform(ink, cv2.DIST_L2, 3); sv=dist[dist>0]
        moments=cv2.moments(ink); angle=0.5*np.degrees(np.arctan2(2*moments.get("mu11",0.0),(moments.get("mu20",0.0)-moments.get("mu02",0.0))+1e-6))
        gaps=[]; run=0
        for v in ink.sum(axis=0):
            if v==0: run+=1
            elif run: gaps.append(run); run=0
        rows=np.where(ink.sum(axis=1)>0)[0]; baseline=float(np.std(rows-rows.mean())/max(1,ink.shape[0])) if rows.size else 0.0
        contours,hierarchy=cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE); holes=int(np.sum(hierarchy[0][:,3]>=0)) if hierarchy is not None else 0
        _,_,stats,_=cv2.connectedComponentsWithStats(ink, connectivity=8); heights=[float(s[3]) for s in stats[1:] if s[4]>10]
        feats=np.array([float(sv.mean()) if sv.size else 0.0,float(sv.std()) if sv.size else 0.0,abs(float(angle)),abs(float(angle))*0.1,float(np.mean(gaps)) if gaps else 0.0,float(np.std(gaps)) if gaps else 0.0,max(0.0,baseline),float(holes/max(1,len(contours))),float(np.mean(heights)) if heights else 0.0,float(np.std(heights)) if heights else 0.0,float(ink.mean()),float(len(heights))],dtype=np.float32)
        feats[~np.isfinite(feats)] = 0.0; np.save(cp, feats); return feats

CENTROIDS={"happy":np.array([1.0,0.6,18.0,1.8,12.0,3.0,0.08,0.18,26.0,4.0,0.18,18.0],dtype=np.float32),"stressed":np.array([2.8,1.4,6.0,3.0,3.0,1.5,0.24,0.08,19.0,6.5,0.35,42.0],dtype=np.float32),"angry":np.array([3.2,1.6,11.0,5.2,5.0,3.8,0.32,0.05,22.0,9.5,0.34,38.0],dtype=np.float32),"calm":np.array([1.3,0.4,4.0,0.6,10.0,1.2,0.05,0.09,24.0,2.5,0.16,16.0],dtype=np.float32),"sad":np.array([1.6,0.8,2.0,1.2,8.0,2.4,0.17,0.05,20.0,4.8,0.14,14.0],dtype=np.float32),"confident":np.array([2.1,0.7,14.0,1.0,9.0,2.0,0.07,0.20,28.0,3.2,0.24,25.0],dtype=np.float32)}
def assign_label(f):
    s,_,sl,_,sp,_,base,loop,ch,chst,ink,_=np.asarray(f,dtype=np.float32)
    if s>2.2 and sp<5 and sl<9: return "stressed",1.0
    if s<1.6 and sp>9 and sl>12: return "happy",1.0
    if base>0.24 and chst>7: return "angry",1.0
    if base<0.08 and sp>8.5 and sl<7: return "calm",1.0
    if ink<0.17 and sl<5 and loop<0.08: return "sad",1.0
    if s>=1.8 and sl>=12 and ch>=24 and base<0.10: return "confident",1.0
    d={k:float(np.linalg.norm(f-v)) for k,v in CENTROIDS.items()}; k,m=min(d.items(), key=lambda x:x[1]); return k,float(max(0.0,min(0.999,1/(1+m/10))))
def generate_labels(data_dir, output_csv, config=None):
    config=config or load_config("config.yaml"); ex=FeatureExtractor(config); rows=[]
    for p in tqdm(list_images(data_dir), desc="Pseudo-labeling"):
        try:
            f=ex.extract(p); label,conf=assign_label(f); rows.append({"image_path":str(Path(p).resolve()),"dataset_name":"cvl" if "cvl" in [x.lower() for x in Path(p).parts] else "iam","label":label,"confidence_score":conf})
        except Exception as e:
            print(f"WARNING: skipping {p}: {e}")
    df=pd.DataFrame(rows); ensure_dir(Path(output_csv).parent); df.to_csv(output_csv,index=False); return df
""")

for rel, content in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
print("part 1 written")
'''))

cells.append(code(r'''
from pathlib import Path
from textwrap import dedent

root = Path(r"C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection")
files = {}

files["src/dataset.py"] = dedent("""
import joblib, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from common import load_config
from feature_label import FeatureExtractor
from preprocessing import preprocess

class GraphologyDataset(Dataset):
    def __init__(self, df, class_to_idx, config, scaler=None):
        self.df=df.reset_index(drop=True).copy(); self.class_to_idx=class_to_idx; self.config=config; self.scaler=scaler; self.extractor=FeatureExtractor(config)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        try:
            img=preprocess(row["image_path"], self.config); feats=self.extractor.extract(row["image_path"])
            if self.scaler is not None: feats=self.scaler.transform(feats.reshape(1,-1)).squeeze(0)
            return torch.tensor(img,dtype=torch.float32), torch.tensor(feats,dtype=torch.float32), int(self.class_to_idx[row["label"]])
        except Exception as e:
            print(f"WARNING: failed sample {row['image_path']}: {e}"); return None
def collate_skip_none(batch):
    batch=[x for x in batch if x is not None]
    if not batch: return None
    a,b,c=zip(*batch); return torch.stack(a), torch.stack(b), torch.tensor(c,dtype=torch.long)
def make_splits(csv_path, config=None, save_scaler_path=None):
    config=config or load_config("config.yaml"); df=pd.read_csv(csv_path); df=df[df["confidence_score"]>=float(config["dataset"]["confidence_threshold"])].reset_index(drop=True)
    names=[n for n in config["project"]["class_names"] if n in sorted(df["label"].unique().tolist())]; class_to_idx={n:i for i,n in enumerate(names)}
    s1=StratifiedShuffleSplit(n_splits=1,test_size=(1.0-config["dataset"]["split_ratios"][0]),random_state=config["dataset"]["random_seed"]); tr_idx,tmp_idx=next(s1.split(df,df["label"]))
    train_df=df.iloc[tr_idx].reset_index(drop=True); tmp=df.iloc[tmp_idx].reset_index(drop=True); rel=config["dataset"]["split_ratios"][2]/(config["dataset"]["split_ratios"][1]+config["dataset"]["split_ratios"][2])
    s2=StratifiedShuffleSplit(n_splits=1,test_size=rel,random_state=config["dataset"]["random_seed"]); va_idx,te_idx=next(s2.split(tmp,tmp["label"]))
    val_df=tmp.iloc[va_idx].reset_index(drop=True); test_df=tmp.iloc[te_idx].reset_index(drop=True)
    ex=FeatureExtractor(config); X=np.stack([ex.extract(p) for p in train_df["image_path"].tolist()]); scaler=StandardScaler().fit(X)
    if save_scaler_path: Path(save_scaler_path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(scaler, save_scaler_path)
    return {"train":GraphologyDataset(train_df,class_to_idx,config,scaler),"val":GraphologyDataset(val_df,class_to_idx,config,scaler),"test":GraphologyDataset(test_df,class_to_idx,config,scaler),"class_to_idx":class_to_idx}
""")

files["src/model_train_eval.py"] = dedent("""
import csv, joblib, numpy as np, psutil, torch, matplotlib.pyplot as plt, pandas as pd, seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet50_Weights, resnet50
from common import ensure_dir, seed_everything
from dataset import collate_skip_none, make_splits

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=6, handcrafted_dim=12, projection_dim=96, hidden=(512,256), dropout=0.35, pretrained=True):
        super().__init__()
        if num_classes < 1 or handcrafted_dim < 1: raise ValueError("invalid dims")
        backbone=resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone=nn.Sequential(*list(backbone.children())[:-1])
        self.feature_stream=nn.Sequential(nn.Linear(handcrafted_dim,projection_dim),nn.ReLU(True),nn.BatchNorm1d(projection_dim),nn.Dropout(dropout/2))
        self.fusion=nn.Sequential(nn.Linear(2048+projection_dim,hidden[0]),nn.ReLU(True),nn.Dropout(dropout),nn.Linear(hidden[0],hidden[1]),nn.ReLU(True),nn.Dropout(dropout/2),nn.Linear(hidden[1],num_classes))
    def logits(self,image,features): return self.fusion(torch.cat([self.backbone(image).flatten(1), self.feature_stream(features)],dim=1))
    def forward(self,image,features): return torch.softmax(self.logits(image,features),dim=1)

def _weights(ds):
    labels=[ds.class_to_idx[x] for x in ds.df["label"].tolist()]; counts=np.bincount(labels)
    return torch.tensor([len(labels)/max(1, counts[i]) for i in range(len(ds.class_to_idx))], dtype=torch.float32)
def _loader(ds,cfg,train=True):
    if train and cfg["training"]["imbalance_strategy"]=="weighted_sampler":
        idxs=[ds.class_to_idx[x] for x in ds.df["label"].tolist()]; counts=np.bincount(idxs); sw=[1.0/max(1,counts[i]) for i in idxs]
        return DataLoader(ds,batch_size=cfg["training"]["batch_size"],sampler=WeightedRandomSampler(sw,len(sw),replacement=True),num_workers=cfg["training"].get("num_workers",2),collate_fn=collate_skip_none)
    return DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=train,num_workers=cfg["training"].get("num_workers",2),collate_fn=collate_skip_none)
def _epoch(model,loader,criterion,optimizer,scaler,device,train=True,clip=1.0,amp=True):
    model.train(train); losses=[]; ok=tot=0
    for batch in loader:
        if batch is None: continue
        img,fts,lab=[x.to(device) for x in batch]
        if train: optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp and device.type=="cuda"):
            logits=model.logits(img,fts); loss=criterion(logits,lab)
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), clip); scaler.step(optimizer); scaler.update()
        pred=logits.argmax(1); losses.append(float(loss.item())); ok += int((pred==lab).sum().item()); tot += int(lab.numel())
    return float(np.mean(losses)) if losses else 0.0, float(ok/max(1,tot))
def train_model(cfg):
    seed_everything(cfg["training"]["random_seed"]); ensure_dir("checkpoints"); ensure_dir(cfg["paths"]["log_dir"])
    splits=make_splits("data/labels.csv", cfg, cfg["paths"]["scaler_path"]); tr,va=splits["train"],splits["val"]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DualStreamModel(len(splits["class_to_idx"]), cfg["model"]["handcrafted_dim"], cfg["model"]["projection_dim"], tuple(cfg["model"]["fusion_hidden_dims"]), cfg["model"]["dropout"], cfg["model"].get("pretrained",True)).to(device)
    criterion=nn.CrossEntropyLoss(weight=_weights(tr).to(device) if cfg["training"]["imbalance_strategy"]=="weighted_loss" else None)
    optimizer=AdamW(model.parameters(), lr=cfg["training"]["learning_rate"]); sched=CosineAnnealingLR(optimizer, T_max=max(1,cfg["training"]["epochs"])); scaler=GradScaler(enabled=cfg["training"]["amp"] and device.type=="cuda")
    with open(Path(cfg["paths"]["log_dir"]) / "training_log.csv","w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(["epoch","train_loss","val_loss","train_acc","val_acc","cpu_mem_mb","gpu_mem_mb"])
    best=float("inf"); wait=0
    for epoch in range(1, int(cfg["training"]["epochs"])+1):
        tl,ta=_epoch(model,_loader(tr,cfg,True),criterion,optimizer,scaler,device,True,cfg["training"]["grad_clip_max_norm"],cfg["training"]["amp"])
        with torch.no_grad(): vl,va_acc=_epoch(model,_loader(va,cfg,False),criterion,optimizer,scaler,device,False,cfg["training"]["grad_clip_max_norm"],cfg["training"]["amp"])
        sched.step(); cpu=psutil.Process().memory_info().rss/(1024**2); gpu=torch.cuda.max_memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0
        with open(Path(cfg["paths"]["log_dir"]) / "training_log.csv","a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([epoch,tl,vl,ta,va_acc,cpu,gpu])
        if vl < best:
            best=vl; wait=0
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scaler_state_dict":scaler.state_dict() if scaler.is_enabled() else None,"val_loss":vl,"val_acc":va_acc,"config_snapshot":cfg,"feature_dim":cfg["feature_extractor"]["feature_dim"],"num_classes":len(splits["class_to_idx"]),"class_names":[k for k,_ in sorted(splits["class_to_idx"].items(), key=lambda x:x[1])]}, cfg["paths"]["best_model"])
        else:
            wait += 1
            if wait >= int(cfg["training"]["patience"]): break
        print(epoch, tl, vl, va_acc)
    return cfg["paths"]["best_model"]
def evaluate_model(cfg):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ck=torch.load(cfg["paths"]["best_model"], map_location=device); scaler=joblib.load(cfg["paths"]["scaler_path"]); splits=make_splits("data/labels.csv", cfg); ds=splits["test"]; ds.scaler=scaler
    model=DualStreamModel(ck["num_classes"], ck["feature_dim"], cfg["model"]["projection_dim"], tuple(cfg["model"]["fusion_hidden_dims"]), cfg["model"]["dropout"], False).to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
    probs=[]; preds=[]; labels=[]
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=False,num_workers=cfg["training"].get("num_workers",2),collate_fn=collate_skip_none):
            if batch is None: continue
            img,fts,lab=[x.to(device) for x in batch]; p=torch.softmax(model.logits(img,fts),dim=1); probs.append(p.cpu().numpy()); preds.append(p.argmax(1).cpu().numpy()); labels.append(lab.cpu().numpy())
    probs=np.concatenate(probs,axis=0); preds=np.concatenate(preds,axis=0); labels=np.concatenate(labels,axis=0); names=ck["class_names"]; out=ensure_dir(cfg["paths"]["evaluation_dir"]); cm=confusion_matrix(labels,preds); cmn=confusion_matrix(labels,preds,normalize="true")
    for m,n,fmt in [(cm,"confusion_matrix_raw.png","d"),(cmn,"confusion_matrix_normalized.png",".2f")]:
        plt.figure(figsize=(8,6)); sns.heatmap(m,annot=True,fmt=fmt,cmap="Blues",xticklabels=names,yticklabels=names); plt.tight_layout(); plt.savefig(Path(out)/n,dpi=200); plt.close()
    report=classification_report(labels,preds,target_names=names,output_dict=True,zero_division=0); rows=[]
    for i,name in enumerate(names):
        try: auc=roc_auc_score((labels==i).astype(int), probs[:,i])
        except ValueError: auc=float("nan")
        rows.append({"class":name,"precision":report[name]["precision"],"recall":report[name]["recall"],"f1":report[name]["f1-score"],"auc_ovr":auc})
    rows.append({"class":"macro_avg","precision":report["macro avg"]["precision"],"recall":report["macro avg"]["recall"],"f1":report["macro avg"]["f1-score"],"auc_ovr":np.nanmean([r["auc_ovr"] for r in rows])})
    pd.DataFrame(rows).to_csv(Path(out)/"metrics_report.csv", index=False)
    if cfg["evaluation"].get("calibration",False):
        plt.figure(figsize=(7,6))
        for i,name in enumerate(names):
            fp,mp=calibration_curve((labels==i).astype(int), probs[:,i], n_bins=cfg["evaluation"].get("calibration_bins",10), strategy="uniform"); plt.plot(mp,fp,marker="o",label=name)
        plt.plot([0,1],[0,1],"--",color="black"); plt.legend(); plt.tight_layout(); plt.savefig(Path(out)/"calibration_plot.png", dpi=200); plt.close()
    return accuracy_score(labels,preds), rows
""")

files["src/inference_nlp.py"] = dedent("""
import joblib, numpy as np, torch
from common import save_json, load_config
from feature_label import FeatureExtractor
from model_train_eval import DualStreamModel
from preprocessing import preprocess
try: import easyocr
except Exception: easyocr=None
try: from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception: SentimentIntensityAnalyzer=None

def _text(path):
    if easyocr is None: return ""
    try: return " ".join(easyocr.Reader(["en"],gpu=False).readtext(path, detail=0)).strip()
    except Exception: return ""
def _sent(text): return {"compound":0.0} if (not text or SentimentIntensityAnalyzer is None) else SentimentIntensityAnalyzer().polarity_scores(text)
def _map(v, probs, names, terms, text):
    p=np.asarray(probs,dtype=np.float32).copy(); c=float(v.get("compound",0.0)); text=text.lower(); terms=[t.lower() for t in terms]
    if -0.05 < c < 0.05: return p
    def boost(label,amount): 
        if label in names: p[names.index(label)] += amount
    if c >= 0.05: boost("happy",0.18); boost("confident",0.10)
    else: boost("angry" if any(t in text for t in terms) else "stressed",0.18); boost("sad",0.08)
    p=np.clip(p,0,None); return p/(p.sum()+1e-8)
def run_inference(image_path, checkpoint_path, scaler_path, output_path, attribution_method="grad", use_nlp=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ck=torch.load(checkpoint_path, map_location=device); scaler=joblib.load(scaler_path); cfg=ck.get("config_snapshot", load_config("config.yaml"))
    model=DualStreamModel(ck["num_classes"], ck["feature_dim"], cfg["model"]["projection_dim"], tuple(cfg["model"]["fusion_hidden_dims"]), cfg["model"]["dropout"], False).to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
    ex=FeatureExtractor(cfg); img=preprocess(image_path,cfg); feats=scaler.transform(ex.extract(image_path).reshape(1,-1)).astype(np.float32)
    imgt=torch.tensor(img,dtype=torch.float32,device=device).unsqueeze(0); ft=torch.tensor(feats,dtype=torch.float32,device=device)
    with torch.no_grad(): probs=torch.softmax(model.logits(imgt,ft),dim=1).cpu().numpy().squeeze(0)
    pred=int(np.argmax(probs)); attrs=None
    if attribution_method=="permutation":
        base=probs[pred]; vals=[]
        for i in range(ft.shape[1]):
            drops=[]; f=ft.detach().cpu().numpy().copy()
            for _ in range(5):
                g=f.copy(); g[0,i]=np.random.permutation(g[0])[i]; gp=torch.tensor(g,dtype=torch.float32,device=device)
                with torch.no_grad(): pr=torch.softmax(model.logits(imgt,gp),dim=1)[0,pred].item()
                drops.append(max(0.0, base-pr))
            vals.append(float(np.mean(drops)))
        attrs=np.array(vals,dtype=np.float32)
    else:
        f2=ft.clone().detach().requires_grad_(True); score=model.logits(imgt,f2)[:,pred].sum(); score.backward(); attrs=f2.grad.detach().cpu().numpy().squeeze(0)
    final=probs.copy(); label=ck["class_names"][pred]; conf=float(final[pred])
    if use_nlp:
        text=_text(image_path); final=((1-cfg["nlp_fusion"]["fusion_weight"])*final)+(cfg["nlp_fusion"]["fusion_weight"]*_map(_sent(text),final,ck["class_names"],cfg["nlp_fusion"].get("high_arousal_terms",[]),text)); final=final/(final.sum()+1e-8); pred=int(np.argmax(final)); label=ck["class_names"][pred]; conf=float(final[pred])
    order=np.argsort(-np.abs(attrs))
    result={"image_path":image_path,"predicted_class":label,"confidence":conf,"attribution_method":attribution_method,"nlp_fusion_applied":bool(use_nlp),"feature_attributions":[{"feature":ex.feature_names[i],"attribution":float(attrs[i]),"rank":int(np.where(order==i)[0][0]+1)} for i in range(len(attrs))],"probabilities":{ck["class_names"][i]:float(final[i]) for i in range(len(ck["class_names"]))}}
    save_json(result, output_path); return result
""")

for rel, content in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
print("part 2 written")
'''))

cells.append(code("""
from pathlib import Path
import pandas as pd

root = Path(r"C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection")
scan = lambda p: sorted(str(x) for x in Path(p).rglob("*") if x.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"})
iam = scan(root / "data/raw/iam/iam_words/words")
cvl = scan(root / "data/raw/cvl")
display(pd.DataFrame([{"dataset":"IAM","num_images":len(iam),"sample":iam[0] if iam else None},{"dataset":"CVL","num_images":len(cvl),"sample":cvl[0] if cvl else None}]))
assert len(iam) + len(cvl) > 0, "Populate data/raw/iam/iam_words/words and/or data/raw/cvl first."
"""))

cells.append(code("""
%cd C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from common import load_config
from feature_label import generate_labels

cfg = load_config("config.yaml")
labels = generate_labels("data/raw", cfg["label_generator"]["output_csv"], cfg)
display(labels.head())
display(labels["label"].value_counts(dropna=False).to_frame("count"))
"""))

cells.append(code("""
%cd C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from common import load_config
from model_train_eval import train_model

cfg = load_config("config.yaml")
train_model(cfg)
"""))

cells.append(code("""
%cd C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from inference_nlp import run_inference

sample = pd.read_csv("data/labels.csv").sample(1, random_state=42)["image_path"].iloc[0]
result = run_inference(sample, "checkpoints/best_model.pt", "checkpoints/feature_scaler.pkl", "output/result.json", attribution_method="grad", use_nlp=False)
result
"""))

cells.append(code("""
%cd C:\\Users\\sherl\\OneDrive\\Desktop\\graphology_ai_emotion_detection
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from common import load_config
from model_train_eval import evaluate_model

cfg = load_config("config.yaml")
accuracy, metrics = evaluate_model(cfg)
print("Test accuracy:", accuracy)
metrics[:2]
"""))

cells.append(md("""
Notes:

- This is a weak-supervision graphology notebook, not a clinically validated emotion detector.
- IAM and CVL are both treated as unlabeled handwriting corpora for pseudo-label bootstrapping.
- If Colab memory is tight, lower `batch_size`, set `pretrained=False`, or replace ResNet50 with ResNet18.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": "Graphology AI Emotion Detection (IAM + CVL).ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("graphology_emotion_detection_colab_iam_cvl.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("graphology_emotion_detection_colab_iam_cvl.ipynb written")
