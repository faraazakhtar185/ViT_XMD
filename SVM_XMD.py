import torch
import numpy as np
from ViT import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#initialize with same params as previous
model = ViT(
    image_size=256,
    patch_size=16,
    num_classes=17, 
    dim=768,
    depth=12, # typically 12
    heads=12, #also typically 12
    mlp_dim=3072, # typically 4×dim
    pool='cls',
    channels=3, # assume rgb
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

state_dict = torch.load('best_vit_weights.pt', map_location=device)
model.load_state_dict(state_dict)

model.eval()
# dataloader for total_data
total_loader   = DataLoader(total_data,   batch_size=64, shuffle=False, num_workers=4)

# now we need to get features and then send them back to the cpu to run our sklearn svm 
# this becomes quite easy because of the change in ViT and I can simply call model.forward_features
all_feats = []
all_labels = []
with torch.no_grad():
    for imgs, labs in total_loader:
        imgs = imgs.to(device)
        feats = model.forward_features(imgs)  # this will give us the feature representation like the original paper 
        all_feats.append(feats.cpu().numpy())   # move back to CPU
        all_labels.append(labs.numpy())

svm_feat = np.vstack(all_feats)      # shape = [N_images, dim] (smaller feature representation than original paper but hopefully richer)
svm_label = np.vstack(all_labels)     # shape = [N_images, 98] -> because we have 98 attributes

# split these into train and val
svm_feat_train, svm_feat_val, svm_label_train, svm_label_val = train_test_split(svm_feat, svm_label, test_size=0.2, random_state=0) #in the paper this split it 12:1

# train SVMs 
svms = []
n_attr = svm_label_train.shape[1]
for i in range(n_attr):
    clf = LinearSVC(
        C=1.0,
        class_weight='balanced',  # helps with rare positives
        max_iter=5000,
    )
    clf.fit(svm_feat_train, svm_label_train[:, i]) #for the particular attribute
    svms.append(clf)

# compute decision scores on train and val
train_scores = np.column_stack([clf.decision_function(svm_feat_train) for clf in svms]) 
test_scores  = np.column_stack([clf.decision_function(svm_feat_val) for clf in svms])  

# average precision score on train and val
# because we have so many cases of values being 0, accuracy would probably be a bad metric for evaluation here
train_aps = average_precision_score(svm_label_train, train_scores, average=None)
test_aps  = average_precision_score(svm_label_val,  test_scores,  average=None)


