import csv
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from server.ml.training import VALUE2CLASSES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
def generate_csv(predictions: dict, img_names:list):
    with open('issues_output.csv', 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'highway']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(img_names)):
            writer.writerow({'image_id': img_names[i], 'highway': predictions[img_names[i]]})


def load_model(path):
    # model = models.mobilenet_v3_small(pretrained=True)
    model = torch.load(path)
    model = model.to(device)
    model.eval()
    # print(model)
    
    return model
def preprocess(img):
    # input
    # print(type(img))
    img = image_transforms(img)
    img = img.unsqueeze(0)

    return img
def get_model(path):
    return load_model(path)
def infer(model, inputs) -> torch.Tensor:
    x = preprocess(inputs)
    
    x = x.to(device)
    
    with torch.no_grad():       
        y_pred = model(x)
        # print(len(y_pred[0]))
        return VALUE2CLASSES[int(y_pred.argmax(dim=1))]

def main():
    # issues_csv_folder_path = input("issues.csv parent folder: ")
    issues_csv_folder_path = "/home/yigit/HackaTUM_Data/dataset/hackatum_dataset/issues"

    # dataset = CustomImageDataset(issues_csv_folder_path, labels_file_name="issues.csv", transform= image_transforms)
    # print(len(dataset), dataset)
    # dataloader  = torch.utils.data.DataLoader(
        # dataset, batch_size=1, num_workers=1
    # )
    # dataset_size = len(dataset)
    # class_names = list(CLASSES.keys())
    import os
    model = get_model("trained_model")
    img_labels = pd.read_csv(os.path.join(issues_csv_folder_path,"issues.csv"))
    dataset_size = len(img_labels.index)
    predictions = {}
    img_names = []
    for i in range(dataset_size):
        classes = img_labels.at[i,"highway"]
        img_name = img_labels.at[i,"image_id"]
        img_full_path = os.path.join(issues_csv_folder_path,f'{img_name}.jpg')
        # print(img_full_path)
                                     
        with Image.open(img_full_path) as img:
            # print(type(img))
            # img = image_transforms(img)
            output = infer(model, img)
            print(classes, output, img_name)
            predictions[img_name] = output
            img_names.append(img_name)
    #print(output)
    generate_csv(predictions,img_names)
    
if __name__ == "__main__":
    main()