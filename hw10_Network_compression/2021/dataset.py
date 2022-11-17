import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

### This block is similar to HW3 ###
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.

train_tfm = transforms.Compose([
  # Resize the image into a fixed shape (height = width = 142)
	transforms.Resize((142, 142)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(15),
	transforms.RandomCrop(128),
	transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 142)
    transforms.Resize((142, 142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

def get_loader(batch_size = 64):
    ### This block is similar to HW3 ###
    # Batch size for training, validation, and testing.
    # A greater batch size usually gives a more stable gradient.
    # But the GPU memory is limited, so please adjust it carefully.

    # Construct datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    
    test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    # Construct data loaders.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, train_set

def get_pseudo_labels(dataset, model, batch_size = 64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    pseudo_labels = []
    for batch in tqdm(loader):
        # A batch consists of image data and corresponding labels.
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))
            pseudo_labels.append(logits.argmax(dim=-1).detach().cpu())
        # Obtain the probability distributions by applying softmax on logits.
    pseudo_labels = torch.cat(pseudo_labels)
    # Update the labels by replacing with pseudo labels.
    for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
        dataset.samples[idx] = (img, pseudo_label.item())
    return dataset

def get_unlabeled_set(): 
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    return unlabeled_set

def semi_dataloader(teacher_net, train_set, batch_size = 64):
    # Generate new trainloader with unlabeled set.
    unlabeled_set = get_unlabeled_set()
    unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net)
    concat_dataset = ConcatDataset([train_set, unlabeled_set])
    train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return train_loader
