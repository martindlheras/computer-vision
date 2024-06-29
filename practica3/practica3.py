import os
import PIL
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

from unet import UNet

# ## Modify mask images
images = os.listdir('data/Image/')
masks = os.listdir('data/Mask/')

image_tensor = list()
masks_tensor = list()

for image in images: 
    dd = PIL.Image.open(f'data/Image/{image}')
    tt = F.pil_to_tensor(dd)
    tt = F.resize(tt, (100, 100))

    tt = tt[None, :, :, :]
    tt = torch.tensor(tt, dtype=torch.float) / 255.

    if tt.shape != (1, 3, 100, 100):
        continue
    image_tensor.append(tt)

    mask = image.replace('.jpg', '.png')
    dd = PIL.Image.open(f'data/Mask/{mask}')
    mm = F.pil_to_tensor(dd)

    mm = mm.repeat(3, 1, 1)
    mm = F.resize(mm, (100, 100))
    mm = mm[:1, :, :]

    mm = torch.tensor((mm > 0.).detach().numpy(), dtype=torch.long)
    mm = torch.nn.functional.one_hot(mm)
    mm = torch.permute(mm, (0, 3, 1, 2))
    mm = torch.tensor(mm, dtype=torch.float)

    masks_tensor.append(mm)

image_tensor = torch.cat(image_tensor)
masks_tensor = torch.cat(masks_tensor)

### New code
split = int(0.8 * len(image_tensor))
train_images = image_tensor[:split]
train_masks = masks_tensor[:split]
test_images = image_tensor[split:]
test_masks = masks_tensor[split:]

# ## Split model
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet(n_channels=3, n_classes=2).to(device)

batch_size = 32
dataloader_train_image = torch.utils.data.DataLoader(train_images, batch_size=batch_size)
dataloader_train_target = torch.utils.data.DataLoader(train_masks, batch_size=batch_size)
dataloader_test_image = torch.utils.data.DataLoader(test_images, batch_size=batch_size)
dataloader_test_target = torch.utils.data.DataLoader(test_masks, batch_size=batch_size)

optim = torch.optim.Adam(unet.parameters(), lr=1e-3)
cross_entropy = torch.nn.CrossEntropyLoss()

loss_list_train = list() 
loss_list_test = list()
jaccard_list_train = list()
jaccard_list_test = list()

for epoch in range(100):
    running_loss_train = 0. 
    running_loss_test = 0. 

    unet.train()
    jaccard_epoch_train = list()

    for image, target in zip(dataloader_train_image, dataloader_train_target):
        # image = image.to(device)
        # target = target.to(device)
        
        optim.zero_grad()
        pred = unet(image)
        
        loss = cross_entropy(pred, target)
        running_loss_train += loss.item()
        
        _, pred_unflatten = torch.max(pred, dim=1)
        _, target_unflatten = torch.max(target, dim=1)

        intersection = torch.sum(pred_unflatten & target_unflatten, dim=(1,2))/torch.sum(pred_unflatten | target_unflatten, dim=(1,2))
        jaccard_epoch_train.append(torch.mean(intersection).detach())
        
        loss.backward()
        optim.step()

    iou_train = sum(jaccard_epoch_train)/len(jaccard_epoch_train)
    jaccard_list_train.append(iou_train)
    loss_list_train.append(running_loss_train)
    
    unet.eval()
    jaccard_epoch_test = list()

    for image, target in zip(dataloader_test_image, dataloader_test_target):
        # image = image.to(device)
        # target = target.to(device)
        
        pred = unet(image)
        
        loss = cross_entropy(pred, target)
        running_loss_test += loss.item()
        
        _, pred_unflatten = torch.max(pred, dim=1)
        _, target_unflatten = torch.max(target, dim=1)

        intersection = torch.sum(pred_unflatten & target_unflatten, dim=(1,2))/torch.sum(pred_unflatten | target_unflatten, dim=(1,2))
        jaccard_epoch_test.append(torch.mean(intersection).detach())

    iou_test = sum(jaccard_epoch_test)/len(jaccard_epoch_test)
    jaccard_list_test.append(iou_test)
    loss_list_test.append(running_loss_test)
    
    print('[Epoch #{}] LossTrain: {:.4f} IoU_Train: {:.4f} LossTest: {:.4f} IoU_Test: {:.4f}'.format(epoch, running_loss_train, iou_train, running_loss_test, iou_test))

# print("Loss list:", loss_list_train)
# print("Loss_test list:", loss_list_test)
# print("Jaccard list:",jaccard_list_train)
# print("Jaccard_test list:",jaccard_list_test)

# Plotting the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_list_train, label='Train Loss')
plt.plot(loss_list_test, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

# Plotting the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(jaccard_list_train, label='Train IoU')
plt.plot(jaccard_list_test, label='Test IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title('IoU')

plt.savefig('split_model.png')
torch.save(unet.state_dict(), 'model_unet_10e.pth')

# ## Full model
unet2 = UNet(n_channels=3, n_classes=2).to(device)

batch_size = 64
dataloader_image = torch.utils.data.DataLoader(image_tensor, batch_size=batch_size)
dataloader_target = torch.utils.data.DataLoader(masks_tensor, batch_size=batch_size)

optim_full = torch.optim.Adam(unet2.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

loss_list = list() 
jaccard_list = list()

for epoch in range(100):
    running_loss = 0. 

    unet2.train()
    jaccard_epoch = list()

    for image, target in zip(dataloader_image, dataloader_target):
        # image = image.to(device)
        # target = target.to(device)
        
        optim_full.zero_grad()
        pred = unet2(image)
        
        loss = criterion(pred, target)
        running_loss += loss.item()
                
        _, pred_unflatten = torch.max(pred, dim=1)
        _, target_unflatten = torch.max(target, dim=1)

        intersection = torch.sum(pred_unflatten & target_unflatten, dim=(1,2))/torch.sum(pred_unflatten | target_unflatten, dim=(1,2))
        jaccard_epoch.append(torch.mean(intersection).detach())
        
        loss.backward()
        optim_full.step()

    iou = sum(jaccard_epoch)/len(jaccard_epoch)
    jaccard_list.append(iou)
    loss_list.append(running_loss)
    
    print('[Epoch #{}] LossTrain: {:.4f} IoU_Train: {:.4f}'.format(epoch, running_loss, iou))

# print("Loss list:", loss_list)
# print("Jaccard list:", jaccard_list)

# Plotting the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss')

# Plotting the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(jaccard_list, label='Train IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title('Train IoU')

plt.savefig('full_model.png')
torch.save(unet.state_dict(), 'model_unet_10e_full.pth')
