import torch
import torch.nn as nn
import time
from torchvision import transforms
import os
from torchsummary import summary
from torch.optim import SGD, Adam
from data_loader import EgoObjectDataset
from triplet_loss import *
from models import Resnet18Triplet
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance
import numpy as np

torch.backends.cudnn.enabled=False

# from losses import TripletLoss, Accuracy

def get_model(pretrained=False, embedding_dimension=512):
    model = Resnet18Triplet(
        embedding_dimension=embedding_dimension,
        pretrained=pretrained
    )
    return model

def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size: batch_size * 2]
    neg_embeddings = embeddings[batch_size * 2:]

    return anc_embeddings, pos_embeddings, neg_embeddings, model

def train():
    image_transforms = transforms.Compose([
        # transforms.RandomApply([transforms.RandomResizedCrop((256,256))], p = 0.2),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ToTensor()
    ])
    learning_rate = 0.075
    batch_size=512
    dataset = EgoObjectDataset('data/final_dataset.csv', transform=image_transforms)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = get_model()
    print(model)
    model = model.cuda()
    model.train()
    margin = 0.2
    l2_distance = PairwiseDistance(p=2)
    progress_bar = enumerate(tqdm(train_dataloader))
    optimizer_model = optim.SGD(
        params=model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        dampening=0,
        nesterov=False,
        weight_decay=1e-5
    )
    criterion = TripletLoss
    total_epochs = 100
    cur_epoch = 0
    while cur_epoch < total_epochs:
        time_now = time.time()
        total_loss = 0
        num_valid_training_triplets = 0
        total_examples = 0
        total_acc = 0
        cur_epoch += 1
        for batch_idx, (batch_sample) in progress_bar:

            # Forward pass - compute embeddings
            anc_imgs = batch_sample[0]
            pos_imgs = batch_sample[1]
            neg_imgs = batch_sample[2]

            # Concatenate the input images into one tensor because doing multiple forward passes would create
            #  weird GPU memory allocation behaviours later on during training which would cause GPU Out of Memory
            #  issues
            all_imgs = torch.cat((anc_imgs, pos_imgs, neg_imgs))  # Must be a tuple of Torch Tensors

            anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                imgs=all_imgs,
                model=model,
                batch_size=batch_size
            )

            pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)

            all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
            valid_triplets = np.where(all == 1)

            anc_valid_embeddings = anc_embeddings[valid_triplets]
            pos_valid_embeddings = pos_embeddings[valid_triplets]
            neg_valid_embeddings = neg_embeddings[valid_triplets]

            triplet_loss = TripletLoss(margin=margin).forward(
                anchor=anc_valid_embeddings,
                positive=pos_valid_embeddings,
                negative=neg_valid_embeddings
            )

            num_valid_training_triplets += len(anc_valid_embeddings)

            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()

        print('Epoch {}:\tNumber of valid training triplets in epoch: {}'.format(
                cur_epoch,
                num_valid_training_triplets
            )
        )



if __name__=='__main__':
    train()