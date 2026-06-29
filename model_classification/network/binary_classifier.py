import sys

import torch
import torch.nn as nn
from .gaitset import SetNet


class BinaryClassificationNet(nn.Module):
    """Binary classification network for depression detection using gait features"""
    
    def __init__(self, dropout_cfg, freeze_cfg, hidden_dim=256):
        super(BinaryClassificationNet, self).__init__()
        
        # 1. The Pretrained Backbone
        self.encoder = SetNet(hidden_dim, dropout_cfg)
        self.hidden_dim = hidden_dim
        
        # 2. Freeze Early Layers to prevent destroying CASIA-B feature extractors
        if freeze_cfg.get('enabled', False):
            self._freeze_early_layers(freeze_cfg)

        # 3. The Bottleneck Classifier Head
        # This replaces the single nn.Linear layer to prevent direct memorization
        self.encoder_p = dropout_cfg.get('head_p', 0.4)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=self.encoder_p), # Single, strategic dropout layer
            nn.Linear(128, 2)
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_early_layers(self, freeze_cfg):
        """Freezes the first few convolutional blocks of the GaitSet backbone."""
        # Freezing the lowest level feature extractors (edges, boundaries, basic shapes)
        layers_to_freeze = []
        encoder_layers_cfg = freeze_cfg.get('encoder_layers', {})
        for layer in encoder_layers_cfg:
            if encoder_layers_cfg[layer]:
                match layer:
                    case "set_layer1":
                        layers_to_freeze.append(self.encoder.set_layer1)
                    case "set_layer2":
                        layers_to_freeze.append(self.encoder.set_layer2)
                    case "set_layer3":
                        layers_to_freeze.append(self.encoder.set_layer3)
                    case "set_layer4":
                        layers_to_freeze.append(self.encoder.set_layer4)
                    case "set_layer5":
                        layers_to_freeze.append(self.encoder.set_layer5)
                    case "set_layer6":
                        layers_to_freeze.append(self.encoder.set_layer6)
                    case "gl_layer1":
                        layers_to_freeze.append(self.encoder.gl_layer1)
                    case "gl_layer2":
                        layers_to_freeze.append(self.encoder.gl_layer2)
                    case "gl_layer3":
                        layers_to_freeze.append(self.encoder.gl_layer3)
                    case "gl_layer4":
                        layers_to_freeze.append(self.encoder.gl_layer4)
                    case "gl_hpm":
                        layers_to_freeze.append(self.encoder.gl_hpm)
                    case "x_hpm":
                        layers_to_freeze.append(self.encoder.x_hpm)
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        #print("--> Early convolutional layers frozen to prevent overfitting.")

    def forward(self, silho, batch_frame=None):
        """
        Args:
            silho: Input silhouette sequences
            batch_frame: Frame batch information
        Returns:
            logits: Classification logits [batch_size, 2]
            features: Extracted features [batch_size, num_bins, hidden_dim]
        """
        # Extract features using the encoder
        features, _ = self.encoder(silho, batch_frame)
        
        # Global Average Pooling across the 62 spatial bins
        gap = torch.mean(features, dim=1) # [batch_size, hidden_dim]
        
        # Apply the bottleneck classification head
        logits = self.classifier(gap)
        
        return logits, features