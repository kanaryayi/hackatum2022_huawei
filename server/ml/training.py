import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

from PIL import Image

# labels_file_name = "labels.csv"  # path of the file

torch.manual_seed(17)

CLASSES = {"primary": 0, "footway": 1}
VALUE2CLASSES = {0: "primary", 1: "footway"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file_name = "labels.csv", transform=None, target_transform=None):
        path = os.path.join(img_dir, labels_file_name)
        print(path)
        self.img_labels = pd.read_csv(path)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = CLASSES

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # print(self.img_labels)
        img_path = os.path.join(
            self.img_dir, f"{str(self.img_labels.at[idx,'image_id'])}" + ".jpg"
        )
        image = Image.open(img_path)
        label = str(self.img_labels.at[idx,"highway"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=48
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # labels = torch.Tensor()
                inputs = inputs.to(device)
                labels = [CLASSES.get(i) for i in labels]
                # print("res", labels)
                labels = torch.Tensor(labels)
                labels = labels.type(torch.uint8)
                # labels =
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(f"labels : {labels} \n preds : {preds}, \noutputs : {outputs}")

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(model)
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = [CLASSES.get(i) for i in labels]
            # print("res", labels)
            labels = torch.Tensor(labels)
            labels = labels.type(torch.uint8)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {VALUE2CLASSES[int(preds[j])]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def save_model(model):
    torch.save(model, "trained_model")


def main():
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.AutoAugment(),
                transforms.TrivialAugmentWide(),
                transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(320),
                # transforms.Random
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    data_dir = "/home/yigit/HackaTUM_Data/dataset/hackatum_dataset"
    image_datasets = {
        x: CustomImageDataset(os.path.join(data_dir, x), transform= data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=16, shuffle=True, num_workers=16
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    # Get a batch of training data

    # model_ft = models.mobilenet_v3_small(pretrained=True, weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    model_ft = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )

    # print(model_ft)
    # print(model_ft.classifier)
    # num_ftrs = model_ft.classifier.in_features

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    print(class_names)
    # print(model_ft)
    # model_ft.fc = nn.Sequential(
    #     nn.Linear(in_features=576,out_features=1024,bias=True),
    #     nn.Hardshrink(),
    #     nn.Dropout(p=0.2, inplace=True),
    #     nn.Linear(in_features=1024, out_features= len(class_names), bias=True))

    model_ft.classifier[3] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    #                             nn.Linear(in_features=512, out_features=len(class_names), bias=True))
    model_ft = model_ft.to(device)
    # print(model_ft)
    # for param in model_ft.parameters():

    for param in model_ft.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.001)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.9)
    # print(model_ft)
    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=48,
    )

    #visualize_model(model_ft, dataloaders)


if __name__ == "__main__":
    main()
