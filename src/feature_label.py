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
