import csv, joblib, numpy as np, psutil, torch, matplotlib.pyplot as plt, pandas as pd, seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from common import ensure_dir, seed_everything
from dataset import collate_skip_none, make_splits

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=6, handcrafted_dim=12, projection_dim=96, hidden=(512,256), dropout=0.35, pretrained=True, backbone_name="resnet50", freeze_backbone=False):
        super().__init__()
        if num_classes < 1 or handcrafted_dim < 1: raise ValueError("invalid dims")
        if backbone_name == "resnet18":
            backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            backbone_dim=512
        else:
            backbone=resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            backbone_dim=2048
        self.backbone=nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.feature_stream=nn.Sequential(nn.Linear(handcrafted_dim,projection_dim),nn.ReLU(True),nn.BatchNorm1d(projection_dim),nn.Dropout(dropout/2))
        self.fusion=nn.Sequential(nn.Linear(backbone_dim+projection_dim,hidden[0]),nn.ReLU(True),nn.Dropout(dropout),nn.Linear(hidden[0],hidden[1]),nn.ReLU(True),nn.Dropout(dropout/2),nn.Linear(hidden[1],num_classes))
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
    model=DualStreamModel(len(splits["class_to_idx"]), cfg["model"]["handcrafted_dim"], cfg["model"]["projection_dim"], tuple(cfg["model"]["fusion_hidden_dims"]), cfg["model"]["dropout"], cfg["model"].get("pretrained",True), cfg["model"].get("backbone","resnet50"), cfg["model"].get("freeze_backbone",False)).to(device)
    criterion=nn.CrossEntropyLoss(weight=_weights(tr).to(device) if cfg["training"]["imbalance_strategy"]=="weighted_loss" else None)
    optimizer=AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["training"]["learning_rate"]); sched=CosineAnnealingLR(optimizer, T_max=max(1,cfg["training"]["epochs"])); scaler=GradScaler(enabled=cfg["training"]["amp"] and device.type=="cuda")
    with open(Path(cfg["paths"]["log_dir"]) / "training_log.csv","w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(["epoch","train_loss","val_loss","train_acc","val_acc","cpu_mem_mb","gpu_mem_mb"])
    best=float("inf"); wait=0
    for epoch in range(1, int(cfg["training"]["epochs"])+1):
        tl,ta=_epoch(model,_loader(tr,cfg,True),criterion,optimizer,scaler,device,True,cfg["training"]["grad_clip_max_norm"],cfg["training"]["amp"])
        with torch.no_grad(): vl,va_acc=_epoch(model,_loader(va,cfg,False),criterion,optimizer,scaler,device,False,cfg["training"]["grad_clip_max_norm"],cfg["training"]["amp"])
        sched.step(); cpu=psutil.Process().memory_info().rss/(1024**2); gpu=torch.cuda.max_memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0
        with open(Path(cfg["paths"]["log_dir"]) / "training_log.csv","a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([epoch,tl,vl,ta,va_acc,cpu,gpu])
        if vl < best:
            best=vl; wait=0
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scaler_state_dict":scaler.state_dict() if scaler.is_enabled() else None,"val_loss":vl,"val_acc":va_acc,"config_snapshot":cfg,"feature_dim":cfg["feature_extractor"]["feature_dim"],"num_classes":len(splits["class_to_idx"]),"class_names":[k for k,_ in sorted(splits["class_to_idx"].items(), key=lambda x:x[1])],"backbone_name":cfg["model"].get("backbone","resnet50")}, cfg["paths"]["best_model"])
        else:
            wait += 1
            if wait >= int(cfg["training"]["patience"]): break
        print(epoch, tl, vl, va_acc)
    return cfg["paths"]["best_model"]
def evaluate_model(cfg):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ck=torch.load(cfg["paths"]["best_model"], map_location=device); scaler=joblib.load(cfg["paths"]["scaler_path"]); splits=make_splits("data/labels.csv", cfg); ds=splits["test"]; ds.scaler=scaler
    model=DualStreamModel(ck["num_classes"], ck["feature_dim"], cfg["model"]["projection_dim"], tuple(cfg["model"]["fusion_hidden_dims"]), cfg["model"]["dropout"], False, ck.get("backbone_name", cfg["model"].get("backbone","resnet50")), False).to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
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
