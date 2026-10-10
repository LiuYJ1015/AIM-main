

import torch
import torch.nn.functional as F
from utils import SLEEPCALoader, SHHSLoader
import numpy as np
import torch.nn as nn
import time
import os
import argparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression as LR
from model import CNNEncoder2D_SLEEP, CNNEncoder2D_SHHS
from loss import OurLoss
from tqdm import tqdm
from collections import Counter
import pickle
from utils import SLEEPCALoader, SHHSLoader
from model import CNNEncoder2D_SLEEP, CNNEncoder2D_SHHS
from utils import read_labels, class_counts, select_indices, stratified_folds

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def task(X_train, X_test, y_train, y_test, n_classes):
    
    cls = LR(solver='lbfgs', multi_class='multinomial', max_iter=500)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    
    res = accuracy_score(y_test, pred)
    
    f1 = f1_score(y_test, pred, average='macro')
    
    return res, f1


def compute_metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=list(range(5)))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(5)),
                                  average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(5)),
                                     average="weighted", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }

def infonce_loss(emb_anchor, emb_positive, T=1.0):
    emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
    emb_positive = F.normalize(emb_positive, p=2, dim=1)
    n = emb_anchor.shape[0]
    emb_total = torch.cat([emb_anchor, emb_positive], dim=0)
    logits = torch.mm(emb_total, emb_total.t())
    idx = torch.arange(2 * n, device=emb_anchor.device)
    logits[idx, idx] = -1e10
    logits = logits / T
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(emb_anchor.device)
    return F.cross_entropy(logits, labels)


def Pretext(encoder, optimizer, Epoch, criterion, pretext_loader, train_loader, test_loader, weight_path):

    encoder.train()


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    for epoch in range(Epoch):
        print (f'\nEpoch: {epoch}')
        epoch_loss =[]

        for index, (aug1, aug2) in enumerate(tqdm(pretext_loader)):
            aug1, aug2 = aug1.to(device), aug2.to(device)
            emb_aug1 = encoder(aug1, mid=False)
            emb_aug2 = encoder(aug2, mid=False)
            loss = criterion(emb_aug1, emb_aug2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())


        mean_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch [{epoch}/{Epoch}] Average Loss: {mean_loss:.4f}")
        scheduler.step(mean_loss)

    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(encoder.state_dict(), weight_path)
    print(f"\n[Save] 预训练完成，模型权重已保存至: {weight_path}")



def evaluate(encoder, train_loader, test_loader, cv_folds=5, seed=1234):
    encoder.eval()

    emb_train, gt_train = [], []
    with torch.no_grad():
        for (X_train, y_train) in train_loader:
            X_train = X_train.to(device)
            emb_train.extend(encoder(X_train).cpu().tolist())
            gt_train.extend(y_train.numpy().flatten())
    emb_train, gt_train = np.array(emb_train), np.array(gt_train)

    emb_test, gt_test = [], []
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            emb_test.extend(encoder(X_test).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())
    emb_test, gt_test = np.array(emb_test), np.array(gt_test)

    cv_summary = {}
    if cv_folds and cv_folds >= 2 and len(emb_train) >= 2:
        folds = stratified_folds(gt_train, cv_folds, seed + 7919)
        fold_rows = []
        for fold_idx, val_local in enumerate(folds):
            train_local = np.setdiff1d(
                np.arange(len(emb_train), dtype=np.int64), val_local, assume_unique=False
            )
            if len(train_local) == 0 or len(val_local) == 0:
                continue
            cls_cv = LR(solver="lbfgs", multi_class="multinomial", max_iter=500)
            cls_cv.fit(emb_train[train_local], gt_train[train_local])
            pred_cv = cls_cv.predict(emb_train[val_local])
            row = compute_metrics(gt_train[val_local], pred_cv)
            row["fold"] = int(fold_idx)
            row["n_train"] = int(len(train_local))
            row["n_val"] = int(len(val_local))
            row["val_class_counts"] = {
                int(c): int(np.sum(gt_train[val_local] == c)) for c in range(5)
            }
            fold_rows.append(row)
        for key in ["acc", "kappa", "macro_f1", "weighted_f1", "balanced_acc"]:
            vals = [r[key] for r in fold_rows if key in r]
            cv_summary[key] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "folds": fold_rows,
            }

    cls = LR(solver="lbfgs", multi_class="multinomial", max_iter=500)
    cls.fit(emb_train, gt_train)
    pred_test = cls.predict(emb_test)
    test_metrics = compute_metrics(gt_test, pred_test)
    try:
        os.makedirs("results", exist_ok=True)
        tag = ("_" + args.tag) if getattr(args, "tag", "") else ""
        pred_path = os.path.join(
            "results",
            f"pred_{args.dataset}_{args.model}{tag}_r{args.train_ratio}_s{seed}.npz",
        )
        np.savez(pred_path, y_true=gt_test, y_pred=pred_test, y_train=gt_train)
        test_metrics["pred_path"] = pred_path
    except Exception as exc:
        print(f"[warn] could not save predictions: {exc}")

    encoder.train()
    return test_metrics, cv_summary

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.5e-3, help="learning rate")
    parser.add_argument('--n_dim', type=int, default=128, help="hidden units (for SHHS, 256, for Sleep, 128)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--pretext', type=int, default=10, help="pretext subject")
    parser.add_argument('--training', type=int, default=10, help="training subject")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--m', type=float, default=0.9995, help="moving coefficient (kept for CLI compatibility; unused with a single shared encoder)")
    parser.add_argument('--model', type=str, default='AIM-STAGE', choices=['AIM-STAGE', 'InfoNCE'], help="objective")
    parser.add_argument('--T', type=float, default=2.0,  help="T")
    parser.add_argument('--sigma', type=float, default=3.0,  help="sigma")
    parser.add_argument('--delta', type=float, default=0.2,  help="delta")
    parser.add_argument('--num_kernels', type=int, default=3, help="number of dynamic Gaussian kernels")
    parser.add_argument('--kernel_bandwidth', type=float, default=0.5, help="Gaussian kernel bandwidth")
    parser.add_argument('--aug_menu', type=str, default='r10', help="augmentation menu (r10 / r10_minus_<op> / r10_<op>_only / none)")
    parser.add_argument('--seed', type=int, default=1234, help="random seed (also selects the labelled subset)")
    parser.add_argument('--tag', type=str, default='', help="optional suffix for result files (used by the ablation driver)")
    parser.add_argument('--dataset', type=str, default='SLEEP', help="dataset")

    parser.add_argument('--phase_mode', type=str, default='sincos_xchan',
                        choices=['sincos_xchan', 'sincos_plus_xchan', 'sincos_lowamp',
                                 'sincos', 'sincos_masked', 'sincos_realimag',
                                 'original', 'angle', 'realimag', 'logpower'],
                        help="STFT phase representation. Default 'sincos_xchan' = channel 1 "
                             "referenced to channel 0 through the inter-channel phase "
                             "difference, encoded as sin/cos (no wrapped angle). Use "
                             "'original' to reproduce the submitted manuscript numbers.")
    parser.add_argument('--no_normalize', action='store_true',
                        help="disable the per-epoch per-channel z-score applied before the STFT")
    
    parser.add_argument('--train_ratio', type=float, default=0.05, help="Ratio of labeled data used for training classifier (e.g. 0.05 for 5 percent)")
    parser.add_argument('--sampling', type=str, default='stratified', choices=['stratified', 'random'],
                        help="low-label sampling strategy: stratified (default) or legacy random")
    parser.add_argument('--min_per_class', type=int, default=1,
                        help="minimum epochs per present class for stratified sampling")
    parser.add_argument('--cv_folds', type=int, default=5,
                        help="cross-validation folds within the sampled labeled subset")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    
    if args.dataset == 'SLEEP':
        pretext_dir = '/data/Docker_Liuyijia2025/AIM-main/preprocess/SLEEP_data/2013/pretext/'
        train_dir = '/data/Docker_Liuyijia2025/AIM-main/preprocess/SLEEP_data/2013/train/'
        test_dir = '/data/Docker_Liuyijia2025/AIM-main/preprocess/SLEEP_data/2013/test/'

        pretext_index = os.listdir(pretext_dir)
        
        train_index_all = sorted(os.listdir(train_dir))
        keep_len = max(1, int(len(train_index_all) * args.train_ratio))
        train_labels = read_labels(train_dir, train_index_all, kind="sleep")
        train_sel = select_indices(
            train_labels, keep_len, seed,
            strategy=args.sampling, min_per_class=args.min_per_class,
        )
        train_index = [train_index_all[i] for i in train_sel]
        print(f"sampling strategy={args.sampling}; selected class counts={class_counts(train_labels, train_sel)}")

        test_index = os.listdir(test_dir)

        print ('pretext (all patient): ', len(pretext_index))
        print (f'train (all patient): {len(train_index)} / {len(train_index_all)} (Ratio: {args.train_ratio})')
        print ('test (all patient): ', len(test_index))

        pretext_loader = torch.utils.data.DataLoader(SLEEPCALoader(pretext_index, pretext_dir, True, aug_menu=args.aug_menu), 
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SLEEPCALoader(train_index, train_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SLEEPCALoader(test_index, test_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        encoder = CNNEncoder2D_SLEEP(args.n_dim, phase_mode=args.phase_mode,
                                     normalize_input=not args.no_normalize).to(device)
    
    elif args.dataset == 'SHHS':
        pretext_dir = '/data/Docker_Liuyijia2025/AIM-main/SHHS_data/processed/pretext/'
        train_dir = '/data/Docker_Liuyijia2025/AIM-main/SHHS_data/processed/train/'
        test_dir = '/data/Docker_Liuyijia2025/AIM-main/SHHS_data/processed/test/'

        pretext_index = os.listdir(pretext_dir)
        
        train_index_all = sorted(os.listdir(train_dir))
        keep_len = max(1, int(len(train_index_all) * args.train_ratio))
        train_labels = read_labels(train_dir, train_index_all, kind="shhs")
        train_sel = select_indices(
            train_labels, keep_len, seed,
            strategy=args.sampling, min_per_class=args.min_per_class,
        )
        train_index = [train_index_all[i] for i in train_sel]
        print(f"sampling strategy={args.sampling}; selected class counts={class_counts(train_labels, train_sel)}")

        test_index = os.listdir(test_dir)

        print ('pretext (all patient): ', len(pretext_index))
        print (f'train (all patient): {len(train_index)} / {len(train_index_all)} (Ratio: {args.train_ratio})')
        print ('test (all patient): ', len(test_index))
        pretext_loader = torch.utils.data.DataLoader(SHHSLoader(pretext_index, pretext_dir, True, aug_menu=args.aug_menu), 
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SHHSLoader(train_index, train_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SHHSLoader(test_index, test_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        encoder = CNNEncoder2D_SHHS(args.n_dim, phase_mode=args.phase_mode,
                                    normalize_input=not args.no_normalize).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model == 'AIM-STAGE':
        criterion = OurLoss(device, args.delta, args.sigma, args.T,
                            kernel_bandwidth=args.kernel_bandwidth,
                            num_kernels=args.num_kernels).to(device)
    else:
        criterion = infonce_loss

    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    weight_name = f"pretrained_{args.dataset}_{args.model}_dim{args.n_dim}.pth"
    weight_path = os.path.join(save_dir, weight_name)

    if os.path.exists(weight_path):
        print(f"\n========== 检测到预训练权重 ==========")
        print(f"[Load] 正在加载权重: {weight_path}")
        encoder.load_state_dict(torch.load(weight_path, map_location=device))
        print("[Skip] 跳过预训练阶段...")
        
        print(f"\n========== 正在评估下游任务 (Train Ratio: {args.train_ratio}) ==========")
        test_metrics, cv_summary = evaluate(
            encoder, train_loader, test_loader,
            cv_folds=args.cv_folds, seed=seed,
        )
        print(f"\n>>> 最终测试集准确率 (Accuracy): {test_metrics['acc']:.4f} <<<")
        print(f">>> 最终测试集 Kappa: {test_metrics['kappa']:.4f} <<<")
        print(f">>> 最终测试集 F1分数 (Macro-F1): {test_metrics['macro_f1']:.4f} <<<")
        print(f">>> 最终测试集 F1分数 (Weighted-F1): {test_metrics['weighted_f1']:.4f} <<<")
        print(f">>> 最终测试集 Balanced Acc: {test_metrics['balanced_acc']:.4f} <<<")
        if cv_summary:
            print(">>> 5-fold CV validation (within sampled X%):")
            for key in ["acc", "kappa", "macro_f1", "weighted_f1", "balanced_acc"]:
                if key in cv_summary:
                    print(f"    cv_{key}: {cv_summary[key]['mean']:.4f} +/- {cv_summary[key]['std']:.4f}")
        print("")
        
    else:
        print(f"\n========== 未检测到预训练权重 ==========")
        print(f"[Train] 将进行 {args.epochs} Epochs 的预训练...")
        Pretext(encoder, optimizer, args.epochs, criterion, pretext_loader, train_loader, test_loader, weight_path)
        
        print(f"\n========== 正在评估下游任务 (Train Ratio: {args.train_ratio}) ==========")
        test_metrics, cv_summary = evaluate(
            encoder, train_loader, test_loader,
            cv_folds=args.cv_folds, seed=seed,
        )
        print(f"\n>>> 最终测试集准确率 (Accuracy): {test_metrics['acc']:.4f} <<<")
        print(f">>> 最终测试集 Kappa: {test_metrics['kappa']:.4f} <<<")
        print(f">>> 最终测试集 F1分数 (Macro-F1): {test_metrics['macro_f1']:.4f} <<<")
        print(f">>> 最终测试集 F1分数 (Weighted-F1): {test_metrics['weighted_f1']:.4f} <<<")
        print(f">>> 最终测试集 Balanced Acc: {test_metrics['balanced_acc']:.4f} <<<")
        if cv_summary:
            print(">>> 5-fold CV validation (within sampled X%):")
            for key in ["acc", "kappa", "macro_f1", "weighted_f1", "balanced_acc"]:
                if key in cv_summary:
                    print(f"    cv_{key}: {cv_summary[key]['mean']:.4f} +/- {cv_summary[key]['std']:.4f}")
        print("")


    end_time = time.time() 
    elapsed_time = end_time - start_time 
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "-" * 30)
    print(f"Total Running Time: {hours}h {minutes}m {seconds}s")
    print("-" * 30)
