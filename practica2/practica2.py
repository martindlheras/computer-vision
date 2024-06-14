from torchvision import transforms, datasets
import torchvision
import torch
import time
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    transforms_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = 'skin_cancer/train'
    test_dir = 'skin_cancer/test'
    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = torchvision.models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    print(num_features)

    model.fc = torch.nn.Linear(num_features, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.fc.parameters(), lr=0.01)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    num_epochs = 25
    start_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {} running'.format(epoch))

        model.train()
        running_loss = 0.
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc * 100,
                                                                           time.time() - start_time))
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for inputs, labels in test_dataloader:
                inputs = inputs.to('cpu')
                labels = labels.to('cpu')

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset)

            test_loss.append(epoch_loss)
            test_accuracy.append(epoch_acc)

            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc * 100,
                                                                             time.time() - start_time))
    torch.save(model.state_dict(), 'model.pth')        
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='train')
    plt.plot(test_accuracy, label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.show()