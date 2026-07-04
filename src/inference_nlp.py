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
    model=DualStreamModel(
        ck["num_classes"],
        ck["feature_dim"],
        cfg["model"]["projection_dim"],
        tuple(cfg["model"]["fusion_hidden_dims"]),
        cfg["model"]["dropout"],
        False,
        ck.get("backbone_name", cfg["model"].get("backbone", "resnet50")),
        False,
    ).to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
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
