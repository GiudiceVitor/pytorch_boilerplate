import torch
import torch.nn.functional as F
import numpy as np
 
class MeanAbsoluteError(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, outputs, labels):
        '''
        Calculates the mean absolute error
 
        Args:
            outputs (_type_): predictions of the model
            labels (_type_): ground truth
 
        Returns:
            loss: torch tensor of the loss containing the loss and the gradients
        '''
 
        return F.l1_loss(outputs, labels)
 

class CategoricalAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, outputs, labels):
        '''
        Calculates the categorical accuracy
 
        Args:
            outputs (_type_): predictions of the model
            labels (_type_): ground truth
 
        Returns:
            accuracy: torch tensor of the accuracy containing the accuracy and the gradients
        '''
       
        outputs = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)
        accuracy = (outputs == labels).float().mean()
        return accuracy