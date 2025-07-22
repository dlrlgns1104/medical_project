import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=1.0, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Unable to read {img_path}. Exception: {e}")
            image = Image.new("RGB", (300, 300), color=(255, 255, 255))
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)

def create_dataset(negative_dir, positive_dir):
    images, labels = [], []
    for fn in os.listdir(negative_dir):
        path = os.path.join(negative_dir, fn)
        if os.path.isfile(path): images.append(path); labels.append(0)
    for fn in os.listdir(positive_dir):
        path = os.path.join(positive_dir, fn)
        if os.path.isfile(path): images.append(path); labels.append(1)
    return np.array(images), np.array(labels, dtype=np.int64)

def plot_confusion_matrix(cm, fold):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative","Positive"],
                yticklabels=["Negative","Positive"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Fold {fold}"); plt.show()

def train_with_kfold(images, labels, n_splits=5, img_size=(300,300), batch_size=16, num_epochs=20):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=30)
    train_transform = A.Compose([
        A.Transpose(p=0.5), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2,0.2,p=0.75),
        A.OneOf([A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.0),
                 A.ElasticTransform(alpha=3)], p=0.9),
        A.CLAHE(4.0,p=0.7), A.HueSaturationValue(10,20,10,p=0.5),
        A.ShiftScaleRotate(0.1,0.1,15,border_mode=0,p=0.85),
        A.Resize(img_size[0], img_size[1]),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2(),
    ])
    all_fold_metrics, all_fprs, all_tprs, all_aucs = [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        train_ds = CustomDataset(images[train_idx], labels[train_idx], transform=train_transform)
        val_ds   = CustomDataset(images[val_idx],   labels[val_idx],   transform=val_transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EfficientNet.from_pretrained('efficientnet-b4')
        model._fc = nn.Sequential(nn.Linear(model._fc.in_features, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64,2))
        model = model.to(device)
        # Focal Loss 사용
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scaler = GradScaler()
        scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=len(train_loader), pct_start=0.3)
        best_val_loss = float('inf'); best_metrics = {}
        for epoch in range(1, num_epochs+1):
            model.train(); running_loss=0.0
            for xb,yb in train_loader:
                xb,yb = xb.to(device), yb.to(device)
                xb,ya,yb_shuf,lam = mixup_data(xb,yb,alpha=1.0,device=device)
                optimizer.zero_grad()
                with autocast(device_type="cuda"): preds=model(xb); loss=mixup_criterion(criterion,preds,ya,yb_shuf,lam)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
                running_loss+=loss.item()
            train_loss = running_loss/len(train_loader)
            model.eval(); val_loss=0.0; all_labels,all_preds,all_probs=[],[],[]
            with torch.no_grad():
                for xb,yb in val_loader:
                    xb,yb = xb.to(device), yb.to(device)
                    with autocast(device_type="cuda"): out=model(xb); loss=criterion(out,yb)
                    val_loss += loss.item()
                    probs=torch.softmax(out,dim=1)[:,1]; preds=out.argmax(dim=1)
                    all_probs.extend(probs.cpu().tolist()); all_preds.extend(preds.cpu().tolist()); all_labels.extend(yb.cpu().tolist())
            val_loss /= len(val_loader); auc=roc_auc_score(all_labels, all_probs)
            print(f"Epoch {epoch}/{num_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  AUC: {auc:.4f}")
            if val_loss<best_val_loss:
                best_val_loss=val_loss; best_metrics={'acc':accuracy_score(all_labels,all_preds),'precision':precision_score(all_labels,all_preds),'recall':recall_score(all_labels,all_preds),'f1':f1_score(all_labels,all_preds),'auc':auc}
                torch.save(model.state_dict(), f"0721_best_model_fold{fold}.pth")
        all_fold_metrics.append({'fold':fold,**best_metrics})
        print(f"Fold {fold} Summary: Acc={best_metrics['acc']:.4f}, Prec={best_metrics['precision']:.4f}, Rec={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}, AUC={best_metrics['auc']:.4f}")
        cm=confusion_matrix(all_labels,all_preds); plot_confusion_matrix(cm, fold)
        fpr,tpr,_=roc_curve(all_labels,all_probs); all_fprs.append(fpr); all_tprs.append(tpr); all_aucs.append(best_metrics['auc'])
        plt.figure();plt.plot(fpr,tpr,label=f"Fold {fold} (AUC={best_metrics['auc']:.4f})");plt.plot([0,1],[0,1],linestyle='--',color='gray');plt.xlabel('FPR');plt.ylabel('TPR');plt.title(f'ROC Curve - Fold {fold}');plt.legend();plt.show()
    mean_metrics={k:np.mean([m[k] for m in all_fold_metrics]) for k in ['acc','precision','recall','f1']}
    mean_metrics['auc']=np.mean(all_aucs)
    plt.figure()
    for fpr,tpr in zip(all_fprs,all_tprs): plt.plot(fpr,tpr,alpha=0.4)
    plt.plot([0,1],[0,1],linestyle='--',color='gray');plt.title(f"Average ROC (Mean AUC = {mean_metrics['auc']:.4f})");plt.xlabel('FPR');plt.ylabel('TPR');plt.show()
    print("\nFinal Metrics Summary:")
    for m in all_fold_metrics: print(f"Fold {m['fold']}: Acc={m['acc']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}")
    print("\nOverall Mean Metrics:")
    for k,v in mean_metrics.items(): print(f"{k.capitalize()}: {v:.4f}")

# 실행
negative_dir = "D:/2024_07_오창교교수님_의료프젝/image_0401/crop comp"
positive_dir = "D:/2024_07_오창교교수님_의료프젝/image_0401/crop incomp"
images, labels = create_dataset(negative_dir, positive_dir)
train_with_kfold(images, labels, n_splits=4, img_size=(224,224), batch_size=16, num_epochs=150) # 0513_2 100 -> 150epoch
