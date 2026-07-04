import warnings

import joblib, numpy as np, pandas as pd, torch
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from pathlib import Path
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

def _drop_rare_classes(df, min_count):
    counts=df["label"].value_counts()
    keep=counts[counts >= min_count].index.tolist()
    dropped=counts[counts < min_count]
    if not dropped.empty:
        warnings.warn(
            "Dropping underrepresented classes before splitting: "
            + ", ".join(f"{label} ({count})" for label, count in dropped.items())
        )
    return df[df["label"].isin(keep)].reset_index(drop=True)

def _cap_samples_per_class(df, max_samples_per_class, random_seed):
    if not max_samples_per_class:
        return df
    pieces=[]
    for _, group in df.groupby("label"):
        pieces.append(group.sample(n=min(len(group), int(max_samples_per_class)), random_state=random_seed))
    capped=pd.concat(pieces, ignore_index=True)
    before=df["label"].value_counts()
    after=capped["label"].value_counts()
    changed=[label for label in after.index if int(after[label]) != int(before.get(label, 0))]
    if changed:
        warnings.warn(
            "Capping samples per class for faster training: "
            + ", ".join(f"{label} ({int(after[label])}/{int(before[label])})" for label in changed)
        )
    return capped

def make_splits(csv_path, config=None, save_scaler_path=None):
    config=config or load_config("config.yaml"); df=pd.read_csv(csv_path); df=df[df["confidence_score"]>=float(config["dataset"]["confidence_threshold"])].reset_index(drop=True)
    df=_drop_rare_classes(df, min_count=2)
    df=_cap_samples_per_class(df, config.get("dataset", {}).get("max_samples_per_class"), config["dataset"]["random_seed"])
    if df.empty:
        raise ValueError("No samples remain after confidence filtering and rare-class removal.")
    names=[n for n in config["project"]["class_names"] if n in sorted(df["label"].unique().tolist())]; class_to_idx={n:i for i,n in enumerate(names)}
    if len(class_to_idx) < 2:
        raise ValueError(f"Need at least 2 classes to train, found {len(class_to_idx)} after filtering.")
    s1=StratifiedShuffleSplit(n_splits=1,test_size=(1.0-config["dataset"]["split_ratios"][0]),random_state=config["dataset"]["random_seed"]); tr_idx,tmp_idx=next(s1.split(df,df["label"]))
    train_df=df.iloc[tr_idx].reset_index(drop=True); tmp=df.iloc[tmp_idx].reset_index(drop=True); rel=config["dataset"]["split_ratios"][2]/(config["dataset"]["split_ratios"][1]+config["dataset"]["split_ratios"][2])
    tmp_counts=tmp["label"].value_counts()
    if int(tmp_counts.min()) >= 2:
        s2=StratifiedShuffleSplit(n_splits=1,test_size=rel,random_state=config["dataset"]["random_seed"]); va_idx,te_idx=next(s2.split(tmp,tmp["label"]))
    else:
        warnings.warn(
            "Validation/test split is falling back to a non-stratified split because some classes "
            "are too rare after the first split: "
            + ", ".join(f"{label} ({count})" for label, count in tmp_counts.items() if count < 2)
        )
        s2=ShuffleSplit(n_splits=1,test_size=rel,random_state=config["dataset"]["random_seed"]); va_idx,te_idx=next(s2.split(tmp))
    val_df=tmp.iloc[va_idx].reset_index(drop=True); test_df=tmp.iloc[te_idx].reset_index(drop=True)
    ex=FeatureExtractor(config); X=np.stack([ex.extract(p) for p in train_df["image_path"].tolist()]); scaler=StandardScaler().fit(X)
    if save_scaler_path: Path(save_scaler_path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(scaler, save_scaler_path)
    return {"train":GraphologyDataset(train_df,class_to_idx,config,scaler),"val":GraphologyDataset(val_df,class_to_idx,config,scaler),"test":GraphologyDataset(test_df,class_to_idx,config,scaler),"class_to_idx":class_to_idx}
