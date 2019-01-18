import torchvision
import torch


def get_dataloader(input_size,batchsize,img_path):
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size,input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    dataset = torchvision.datasets.ImageFolder(img_path, transform=transforms)
    dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batchsize,
    shuffle=True,
    drop_last=True,
    )
    return dataloader