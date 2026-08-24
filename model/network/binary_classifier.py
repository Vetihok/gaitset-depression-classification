import sys

import torch
import torch.nn as nn

from .gaitset import SetNet


class BinaryClassificationNet(nn.Module):
    """Binary classification network for depression detection using gait features"""
    
    def __init__(self, dropout_cfg, freeze_cfg, hidden_dim=256, classifier_head_cfg=None):
        super(BinaryClassificationNet, self).__init__()
        
        # 1. The Pretrained Backbone
        self.encoder = SetNet(hidden_dim, dropout_cfg)
        self.hidden_dim = hidden_dim
        
        # 2. Freeze Early Layers to prevent destroying CASIA-B feature extractors
        if freeze_cfg.get('enabled', False):
            self._freeze_early_layers(freeze_cfg)

        # 3. The Bottleneck Classifier Head
        # This replaces the single nn.Linear layer to prevent direct memorization
        self.head_p = dropout_cfg.get('head_p', 0.4)
        self.classifier = self._build_classifier_head(classifier_head_cfg)
        if freeze_cfg.get("freeze_classifier", False):
            self.freeze_classifier()

        self._initialize_weights()

    def _build_classifier_head(self, classifier_head_cfg):
        if not classifier_head_cfg:
            return nn.Sequential(nn.Linear(self.hidden_dim, 2))

        layer_defs = classifier_head_cfg.get('layers', []) if isinstance(classifier_head_cfg, dict) else classifier_head_cfg
        if not isinstance(layer_defs, list) or not layer_defs:
            return nn.Sequential(nn.Linear(self.hidden_dim, 2))

        layers = []
        in_features = self.hidden_dim
        for layer_cfg in layer_defs:
            if not isinstance(layer_cfg, dict):
                raise TypeError("Each classifier head layer must be defined as a dictionary")

            layer_type = layer_cfg.get('type', 'linear').lower()
            if layer_type == 'linear':
                out_features = layer_cfg.get('out_features', 2)
                layers.append(nn.Linear(in_features, out_features))
                in_features = out_features
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
            elif layer_type == 'leaky_relu':
                negative_slope = layer_cfg.get('negative_slope', 0.1)
                layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif layer_type == 'dropout':
                p = layer_cfg.get('p', self.head_p)
                layers.append(nn.Dropout(p=p))
            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_type == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported classifier head layer type: {layer_type}")

        if not layers:
            return nn.Sequential(nn.Linear(self.hidden_dim, 2))
        return nn.Sequential(*layers)
    
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
            if encoder_layers_cfg[layer] or freeze_cfg.get('freeze_whole_encoder', False):
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

    def unfreeze_encoder(self):
        """Unfreeze every layers of the encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                for param in m.parameters():
                    param.requires_grad = False

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