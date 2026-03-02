import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from dataset import Cifar10Dataset
from highwaynetwork import HighwayNetwork

import wandb


class HighwayNetworkModel:
    def __init__(self, device,log=True):
        self.model = HighwayNetwork().to(device)
        self.device = device
        self.log = log

    def train(self, root_directory, hyperparams, spatial_transforms, color_transforms, validation_transforms):
        if self.log:
            wandb_log = self.init_logging(hyperparams.batch_size, hyperparams.learning_rate, hyperparams.momentum, hyperparams.weight_decay, hyperparams.num_epochs)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=hyperparams.learning_rate, momentum=hyperparams.momentum, weight_decay=hyperparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
        
        # Create Datasets 
        training_dataset = Cifar10Dataset(root_directory, dataset="train.csv", transform=spatial_transforms)
        train_dataloader = DataLoader(training_dataset, batch_size=hyperparams.batch_size, shuffle=True)
        
        val_dataset = Cifar10Dataset(root_directory, dataset="test.csv", transform=validation_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False)

        best_loss = float("inf")
        logging_steps = 0
        for epoch in range(hyperparams.num_epochs):

            # begin training loop 
            self.model.train()
            running_loss = 0.0
            for i, (input, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                assert self.model.training, 'make sure the network is in train mode with `.train()`'

                optimizer.zero_grad() # zero out the gradient from previous batches
                
                input = input.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input) # Predictions

                loss = F.cross_entropy(outputs, labels)

                loss.backward() # backprop step
                optimizer.step() # SGD optimizing step

                running_loss += loss.item()
                if self.log and (i % 10 == 0) and i > 0:
                    wandb_log.log({"eval/loss": loss.item()}, step=logging_steps)
                    logging_steps += 1
            
            validation_loss, validation_accuracy, validation_top_5_accuracy = self.validation(val_dataset, val_dataloader) 
            scheduler.step(validation_loss) # perform lr reduction if validation loss doesn't improve

            print(f"Epoch: {epoch+1} Training Loss: {running_loss/len(train_dataloader)}")
            print(f" Validation Loss: {validation_loss} Validation Accuracy: {validation_accuracy} Validation Top 5 Accuracy {validation_top_5_accuracy}")
            if self.log:
                wandb_log.log({
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_top_5_accuracy": validation_top_5_accuracy,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            if validation_loss < best_loss:
                torch.save(self.model.state_dict(), "model_weights/best_val_loss.pt")
                best_loss = validation_loss

    def validation(self, val_dataset, val_dataloader):
        total_loss = 0
        correct = 0
        top_5_correct = 0

        self.model.eval() # set model to eval mode so the weights won't get changed
        with torch.no_grad():
            for input, labels in tqdm(val_dataloader, desc="Validation Run"):
                input = input.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input) 

                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                # get accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                top_5_preds = torch.topk(outputs, 5, dim=1).indices
                top_5_correct += (top_5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        avg_loss = total_loss / len(val_dataloader)  
        accuracy = correct / len(val_dataset)
        top_5_accuracy = top_5_correct / len(val_dataset)

        return avg_loss, accuracy, top_5_accuracy
    
    def init_logging(self, batch_size, learning_rate, momentum, weight_decay, num_epochs):
        wandb_log = wandb.init(
            entity="wtoth21",
            # Set the wandb project where this run will be logged.
            project="HighwayNetwork",
            # Track hyperparameters and run metadata.
            config={
                "architecture": "Highway Network",
                "dataset": "Cifar10",
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "momentum": momentum,
                "l2 regularization": weight_decay,
                "epochs": num_epochs,
            })
        return wandb_log