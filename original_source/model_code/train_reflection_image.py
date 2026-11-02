"""Train an image-only paired ViT on binary reflection masks."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import IMG_MEAN, IMG_STD
from dataset_reflection_images import prepare_dataset
from model_reflection_image_pair import ReflectionImagePairClassifier


class MaskPairDataset(Dataset):
    def __init__(self, images, records, split):
        self.images = images
        self.records = [r for r in records if r["split"] == split]
        self.indices = [records.index(r) for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        image = torch.from_numpy(self.images[self.indices[i]].copy()).float().unsqueeze(1).repeat(1, 3, 1, 1) / 255.0
        mean = torch.tensor(IMG_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMG_STD).view(1, 3, 1, 1)
        image = (image - mean) / std
        record = self.records[i]
        return {"images": image, "labels": torch.tensor(record["labels"], dtype=torch.float32),
                "stratum": torch.tensor(record["stratum"], dtype=torch.long), "name": record["name"]}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (((1 - torch.exp(-bce)) ** self.gamma) * bce).mean()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def ranking_loss(logits, labels):
    unilateral = labels.sum(1) == 1
    if not unilateral.any(): return logits.new_zeros(())
    z, y = logits[unilateral], labels[unilateral]
    weak = torch.where(y[:, 0] > .5, z[:, 0] - z[:, 1], z[:, 1] - z[:, 0])
    return torch.relu(.5 - weak).mean()


def metrics(logits, labels, threshold=.5):
    score = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    truth = labels.detach().cpu().numpy().reshape(-1).astype(int)
    pred = (score >= threshold).astype(int)
    return {"threshold": threshold, "acc": float(accuracy_score(truth, pred)),
            "balanced_acc": float(balanced_accuracy_score(truth, pred)),
            "prec": float(precision_score(truth, pred, zero_division=0)),
            "rec": float(recall_score(truth, pred, zero_division=0)),
            "f1": float(f1_score(truth, pred, zero_division=0)),
            "auc": float(roc_auc_score(truth, score)),
            "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]).tolist()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); logits_all=[]; labels_all=[]; strata=[]
    for batch in loader:
        if "tokens" in batch:
            logits = model.forward_from_tokens(batch["tokens"].to(device))
        else:
            logits = model(batch["images"].to(device))
        logits_all.append(logits.cpu()); labels_all.append(batch["labels"]); strata.append(batch["stratum"])
    logits=torch.cat(logits_all); labels=torch.cat(labels_all); strata=torch.cat(strata)
    base=metrics(logits, labels)
    unilateral=(strata==2)|(strata==3)
    if unilateral.any():
        predicted=torch.argmax(logits[unilateral],1).numpy(); truth=(labels[unilateral,1]>.5).numpy()
        base["laterality_n"]=int(unilateral.sum()); base["laterality_acc"]=float((predicted==truth).mean())
    else: base["laterality_n"]=0; base["laterality_acc"]=None
    return base


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=46); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--scope",choices=["global","macula"],default="global"); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--accum",type=int,default=8); ap.add_argument("--lr",type=float,default=1e-4); args=ap.parse_args()
    seed_everything(args.seed); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    out=Path(f"/root/hospital/model_code/outputs/reflection_image_only_{args.scope}_seed{args.seed}"); data_dir=out/"data"
    manifest=data_dir/"manifest.json"
    if not manifest.exists(): prepare_dataset("/root/the_dataset.csv",data_dir,scope=args.scope)
    payload=json.loads(manifest.read_text(encoding="utf-8")); records=payload["records"]; images=np.load(data_dir/"images_uint8.npy",mmap_mode="r")
    train=MaskPairDataset(images,records,"train"); val=MaskPairDataset(images,records,"val"); test=MaskPairDataset(images,records,"test")
    weights=np.array([1.0 if r["stratum"]==0 else 3.0 for r in train.records],dtype=np.float64)
    sampler=WeightedRandomSampler(torch.tensor(weights),len(weights),replacement=True,generator=torch.Generator().manual_seed(args.seed))
    train_loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=0); val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False); test_loader=DataLoader(test,batch_size=args.batch_size,shuffle=False)
    model=ReflectionImagePairClassifier().to(device)
    # Prefix computation is frozen; only the final ViT block, projection, pair, and heads train.
    model.eval(); cached={}
    with torch.no_grad():
        for split_ds in (train,val,test):
            for start in range(0,len(split_ds),args.batch_size):
                batch=torch.stack([split_ds[i]["images"] for i in range(start,min(start+args.batch_size,len(split_ds)))])
                b=batch.shape[0]; cached[split_ds.records[start]["split"]]=cached.get(split_ds.records[start]["split"],[])
                tokens = model.frozen_prefix(batch.reshape(b*2,3,384,384).to(device)).cpu().reshape(b, 2, -1, 768)
                cached[split_ds.records[start]["split"]].extend(tokens)
    # Replace loaders with token datasets to avoid recomputing the frozen ViT prefix.
    class TokenDataset(Dataset):
        def __init__(self, ds): self.ds=ds; self.tokens=cached[ds.records[0]["split"]]
        def __len__(self): return len(self.ds)
        def __getitem__(self,i): return {"tokens":self.tokens[i].reshape(2,*self.tokens[i].shape[1:]),"labels":self.ds[i]["labels"],"stratum":self.ds[i]["stratum"]}
    train_t,val_t,test_t=TokenDataset(train),TokenDataset(val),TokenDataset(test)
    train_loader=DataLoader(train_t,batch_size=args.batch_size,sampler=sampler); val_loader=DataLoader(val_t,batch_size=args.batch_size); test_loader=DataLoader(test_t,batch_size=args.batch_size)
    criterion=FocalLoss(); opt=AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4); best=-1; best_state=None; patience=0; history=[]
    def forward_tokens(batch): return model.forward_from_tokens(batch["tokens"].to(device))
    for epoch in range(1,args.epochs+1):
        model.train(); opt.zero_grad(); total=0
        for step,batch in enumerate(train_loader,1):
            logits=forward_tokens(batch); loss=criterion(logits,batch["labels"].to(device))+ranking_loss(logits,batch["labels"].to(device)); (loss/args.accum).backward(); total+=loss.item()
            if step%args.accum==0 or step==len(train_loader): torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); opt.zero_grad()
        model.eval(); vals=[]; vlabels=[]
        with torch.no_grad():
            for batch in val_loader: vals.append(forward_tokens(batch).cpu()); vlabels.append(batch["labels"])
        vm=metrics(torch.cat(vals),torch.cat(vlabels)); history.append({"epoch":epoch,"train_loss":total/len(train_loader),**vm}); print(f"Epoch {epoch:02d} loss={total/len(train_loader):.4f} val_auc={vm['auc']:.4f} val_f1={vm['f1']:.4f}",flush=True)
        if vm["auc"]>best: best=vm["auc"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=10: break
    model.load_state_dict(best_state); test_metrics=evaluate(model,test_loader,device)
    result={"scope":args.scope,"seed":args.seed,"model":"ReflectionImagePairClassifier","inputs":"two binary reflection images only","input_shape":["B",2,3,384,384],"patients":len(records),"split_counts":payload["report"]["split_counts"],"parameters":sum(p.numel() for p in model.parameters()),"best_val_auc":best,"test_metrics":test_metrics,"history":history,"data_manifest":str(manifest)}
    out.mkdir(parents=True,exist_ok=True); (out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); torch.save(model.state_dict(),out/"best_model.pt"); print(json.dumps(result["test_metrics"],ensure_ascii=False,indent=2))


if __name__=="__main__": main()
