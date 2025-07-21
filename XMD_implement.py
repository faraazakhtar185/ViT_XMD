import torch
from ViT import *
from torch.utils.data import DataLoader


#I dont have any data, so here I will just assume that this data exists and that it is structured as image, label.
# The labels have 17 colums each with a value in [0,1], denoting whether the attribute is present in the image. 
#I will also assume that I have split the data into "train" and "val". 
train_loader = DataLoader(train, batch_size=64, shuffle=True,  num_workers=4) #shuffle on train and some gpu optimization
val_loader   = DataLoader(val,   batch_size=64, shuffle=False, num_workers=4) #no shuffle and some gpu optimization


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViT(
    image_size=256,
    patch_size=16,
    num_classes=17, #according to cnn paper, we can do the svm stuff later
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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

#train for 10 epochs
epochs = 10
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)               
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()    
        running_loss += loss.item() * imgs.size(0) 
        epoch_train_loss = running_loss / len(train)


#evaluation 
    model.eval()
    correct = 0
    total   = 0
    val_loss = 0.0
    best_val_loss = float('inf')

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

    epoch_val_loss = val_loss / len(val)
    val_acc = correct / total # validation accuracy
    # save best model so we can load it later
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_vit_weights.pt')