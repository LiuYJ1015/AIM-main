

import torch
from utils import SLEEPCALoader, SHHSLoader
import numpy as np
import torch.nn as nn
import time
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from loss import MoCo, BYOL, OurLoss, SimSiam
from tqdm import tqdm
from collections import Counter
import pickle
from model import CNNEncoder2D_SLEEP, CNNEncoder2D_SHHS

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# evaluation design
def task(X_train, X_test, y_train, y_test, n_classes):
    
    cls = LR(solver='lbfgs', multi_class='multinomial', max_iter=500)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    
    res = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred) 
    return res, cm

def Pretext(q_encoder, k_encoder, optimizer, Epoch, criterion, pretext_loader, train_loader, test_loader, weight_path):

    q_encoder.train(); k_encoder.train()

    global queue
    global queue_ptr
    global n_queue

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    for epoch in range(Epoch):
        print (f'\nEpoch: {epoch}')
        epoch_loss =[] # 记录当前 epoch 的所有 loss

        for index, (aug1, aug2) in enumerate(tqdm(pretext_loader)):
            aug1, aug2 = aug1.to(device), aug2.to(device)
            if args.model in ['BYOL']:
                emb_aug1 = q_encoder(aug1, mid=False, byol=True)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['AIM']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['MoCo']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['SimSiam']:
                emb_aug1, proj1 = q_encoder(aug1, simsiam=True)
                emb_aug2, proj2 = q_encoder(aug2, simsiam=True)

            # backpropagation
            if args.model == 'MoCo':
                loss = criterion(emb_aug1, emb_aug2, queue)
                if queue_ptr + emb_aug2.shape[0] > n_queue:
                    queue[queue_ptr:] = emb_aug2[:n_queue-queue_ptr]
                    queue[:queue_ptr+emb_aug2.shape[0]-n_queue] = emb_aug2[-(queue_ptr+emb_aug2.shape[0]-n_queue):]
                    queue_ptr = (queue_ptr + emb_aug2.shape[0]) % n_queue
                else:
                    queue[queue_ptr:queue_ptr+emb_aug2.shape[0]] = emb_aug2
            elif args.model == 'SimSiam':
                loss = criterion(proj1, proj2, emb_aug1, emb_aug2)
            else:
                loss = criterion(emb_aug1, emb_aug2)

            # loss back
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # only update encoder_q
            epoch_loss.append(loss.item())

            # exponential moving average (EMA)
            for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
                param_k.data = param_k.data * args.m + param_q.data * (1. - args.m) 

        mean_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch [{epoch}/{Epoch}] Average Loss: {mean_loss:.4f}")
        scheduler.step(mean_loss)

    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(q_encoder.state_dict(), weight_path)



def evaluate(q_encoder, train_loader, test_loader):

    # freeze
    q_encoder.eval()

    # process val
    emb_val, gt_val = [],[]
    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val).cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)

    emb_test, gt_test = [],[]
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())
    emb_test, gt_test= np.array(emb_test), np.array(gt_test)

  
    res, cm = task(emb_val, emb_test, gt_val, gt_test, 5)
    
    q_encoder.train()
    return res

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.5e-3, help="learning rate")
    parser.add_argument('--n_dim', type=int, default=128, help="hidden units (for SHHS, 256, for Sleep, 128)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--pretext', type=int, default=10, help="pretext subject")
    parser.add_argument('--training', type=int, default=10, help="training subject")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--m', type=float, default=0.9995, help="moving coefficient")
    parser.add_argument('--model', type=str, default='AIM', help="which model")
    parser.add_argument('--T', type=float, default=0.3,  help="T")
    parser.add_argument('--sigma', type=float, default=2.0,  help="sigma")
    parser.add_argument('--delta', type=float, default=0.2,  help="delta")
    parser.add_argument('--dataset', type=str, default='SLEEP', help="dataset")
    
    parser.add_argument('--train_ratio', type=float, default=0.05, help="Ratio of labeled data used for training classifier (e.g. 0.05 for 5%)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    # set random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    global queue
    global queue_ptr
    global n_queue
    
    if args.dataset == 'SLEEP':
        # dataset
        pretext_dir = '/data/Docker_Liuyijia2025/ContraWR-main/TEST/2013/pretext/'
        train_dir = '/data/Docker_Liuyijia2025/ContraWR-main/TEST/2013/train/'
        test_dir = '/data/Docker_Liuyijia2025/ContraWR-main/TEST/2013/test/'

        pretext_index = os.listdir(pretext_dir)
        
        train_index_all = os.listdir(train_dir)
        train_index_all.sort() 
        np.random.shuffle(train_index_all) 
        
        keep_len = max(1, int(len(train_index_all) * args.train_ratio))
        train_index = train_index_all[:keep_len]

        test_index = os.listdir(test_dir)

        print ('pretext (all patient): ', len(pretext_index))
        print (f'train (all patient): {len(train_index)} / {len(train_index_all)} (Ratio: {args.train_ratio})')
        print ('test (all patient): ', len(test_index))

        pretext_loader = torch.utils.data.DataLoader(SLEEPCALoader(pretext_index, pretext_dir, True), 
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SLEEPCALoader(train_index, train_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SLEEPCALoader(test_index, test_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        q_encoder = CNNEncoder2D_SLEEP(args.n_dim).to(device)
        k_encoder = CNNEncoder2D_SLEEP(args.n_dim).to(device)
    
    elif args.dataset == 'SHHS':
        # dataset
        pretext_dir = '/data/Docker_Liuyijia2025/AIM-main/SHHS/processed/pretext/'
        train_dir = '/data/Docker_Liuyijia2025/AIM-main/SHHS/processed/train/'
        test_dir = '/data/Docker_Liuyijia2025/AIM-main/SHHS/processed/test/'

        pretext_index = os.listdir(pretext_dir)
        
     
        train_index_all = os.listdir(train_dir)
        train_index_all.sort() 
        np.random.shuffle(train_index_all) 
        
        keep_len = max(1, int(len(train_index_all) * args.train_ratio))
        train_index = train_index_all[:keep_len]
      

        test_index = os.listdir(test_dir)

        print ('pretext (all patient): ', len(pretext_index))

        print (f'train (all patient): {len(train_index)} / {len(train_index_all)} (Ratio: {args.train_ratio})')
        print ('test (all patient): ', len(test_index))
        pretext_loader = torch.utils.data.DataLoader(SHHSLoader(pretext_index, pretext_dir, True), 
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SHHSLoader(train_index, train_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SHHSLoader(test_index, test_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        # define the model
        q_encoder = CNNEncoder2D_SHHS(args.n_dim).to(device)
        k_encoder = CNNEncoder2D_SHHS(args.n_dim).to(device)

    for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        param_k.data.copy_(param_q.data) 
        param_k.requires_grad = False  # not update by gradient

    optimizer = torch.optim.Adam(q_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # assign contrastive loss function
    if args.model == 'AIM':
        criterion = OurLoss(device, args.delta, args.sigma, args.T).to(device)
    elif args.model == 'MoCo':
        criterion = MoCo(device).to(device)
        queue_ptr, n_queue = 0, 4096
        queue = torch.tensor(np.random.rand(n_queue, args.n_dim), dtype=torch.float).to(device)
    elif args.model == 'BYOL':
        criterion = BYOL(device).to(device)
    elif args.model == 'SimSiam':
        criterion = SimSiam(device).to(device)

    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    weight_name = f"pretrained_{args.dataset}_{args.model}_dim{args.n_dim}.pth"
    weight_path = os.path.join(save_dir, weight_name)

    if os.path.exists(weight_path):
        q_encoder.load_state_dict(torch.load(weight_path, map_location=device))
        
        final_acc = evaluate(q_encoder, train_loader, test_loader)
  
        
    else:
   
        Pretext(q_encoder, k_encoder, optimizer, args.epochs, criterion, pretext_loader, train_loader, test_loader, weight_path)
        
        final_acc= evaluate(q_encoder, train_loader, test_loader)
        print(f"\n>>>  (Accuracy): {final_acc:.4f} <<<")
    

    end_time = time.time() 
    elapsed_time = end_time - start_time 
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "-" * 30)
    print(f"Total Running Time: {hours}h {minutes}m {seconds}s")
    print("-" * 30)
