import sys

import torch
import torch.nn as nn
from .gaitset import SetNet


class BinaryClassificationNet(nn.Module):
    """Binary classification network for depression detection using gait features"""
    
    def __init__(self, hidden_dim =256):
        super(BinaryClassificationNet, self).__init__()
        #print(f"--BinaryClassificationNet SetNet:\tinput=--\toutput=--\t--\t--")
        self.encoder = SetNet(hidden_dim)
        self.hidden_dim = hidden_dim
        #self.batch_size = batch_size
        # Classification head: 2 output classes (Normal=0, Depressed=1)
        self.classifier = nn.Linear(self.hidden_dim, 2)
        #print(f"--BinaryClassificationNet Linear:\tinput={hidden_dim}\toutput={2}\t--\t--")
    
    def forward(self, silho, batch_frame=None):
        """
        Args:
            silho: Input silhouette sequences
            batch_frame: Frame batch information
        Returns:
            logits: Classification logits [batch_size, 2]
            features: Extracted features [batch_size, num_bins, hidden_dim]
        """
        # Extract features using the encoder (returns tuple: feature, None)
        features, _ = self.encoder(silho, batch_frame)
        gap = torch.mean(features, dim=[1])
        # features shape: [batch_size, num_bins, hidden_dim] where num_bins=62
        # Get actual batch size from the tensor, not from initialization
        actual_batch_size = features.shape[0]
        #features_pooled = features.view(actual_batch_size, -1)  # [batch_size, num_bins*hidden_dim]
        # Apply classification head
        logits = self.classifier(gap)
        
        return logits, features

