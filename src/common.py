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
