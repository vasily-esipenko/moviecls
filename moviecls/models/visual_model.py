import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch


class VisualEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=300, backbone="resnet50", pretrained_weights=None, freeze_layers="none"):
        super(VisualEmbeddingModel, self).__init__()

        if backbone == "resnet50":
            self.backbone = models.resnet50(weights='IMAGENET1K_V2')
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet18":
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone.startswith("efficientnet_b0"):
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b7":
            self.backbone = models.efficientnet_b7(weights='IMAGENET1K_V1')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "vit_b_16":
            self.backbone = models.vit_b_16(weights='IMAGENET1K_V1')
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
        elif backbone == "vit_l_16":
            self.backbone = models.vit_l_16(weights='IMAGENET1K_V1')
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
        else:
            raise ValueError(f"Неподдерживаемая архитектура: {backbone}")

        if pretrained_weights:
            self.backbone.load_state_dict(torch.load(pretrained_weights))
            print(
                f"Загружены пользовательские веса для {backbone} из {pretrained_weights}")

        self._freeze_layers(backbone, freeze_layers)

        self.projection = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def _freeze_layers(self, backbone, freeze_mode):
        if freeze_mode == "none":
            return

        elif freeze_mode == "all":
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Заморожены все слои в {backbone}")

        elif freeze_mode == "all_except_last":
            if backbone.startswith("resnet"):
                for name, param in self.backbone.named_parameters():
                    if "layer4" not in name:
                        param.requires_grad = False
                print(
                    f"Заморожены все слои кроме последнего блока в {backbone}")

            elif backbone.startswith("efficientnet"):
                total_blocks = len(self.backbone.features)
                for i, layer in enumerate(self.backbone.features):
                    if i < total_blocks - 2:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(
                    f"Заморожены все слои кроме 2 последних блоков в {backbone}")

            elif backbone.startswith("vit"):
                for name, param in self.backbone.named_parameters():
                    if "embeddings" in name or "encoder.layers.0." in name or "encoder.layers.1." in name:
                        param.requires_grad = False
                print(
                    f"Заморожены эмбеддинги и первые слои трансформера в {backbone}")

        elif freeze_mode == "partial":
            if backbone.startswith("resnet"):
                for name, param in self.backbone.named_parameters():
                    if "layer3" not in name and "layer4" not in name:
                        param.requires_grad = False
                print(f"Заморожены ранние слои (до layer3) в {backbone}")

            elif backbone.startswith("efficientnet"):
                total_blocks = len(self.backbone.features)
                for i, layer in enumerate(self.backbone.features):
                    if i < total_blocks // 2:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"Заморожена первая половина блоков в {backbone}")

            elif backbone.startswith("vit"):
                for name, param in self.backbone.named_parameters():
                    if "embeddings" in name:
                        param.requires_grad = False
                print(f"Заморожены только эмбеддинги в {backbone}")

        trainable_params = sum(p.numel()
                               for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(
            f"Обучаемые параметры: {trainable_params:,} из {total_params:,} ({trainable_params/total_params:.2%})")

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def create_model(config, device):
    model = VisualEmbeddingModel(
        embedding_dim=config.model.embedding_dim,
        backbone=config.model.backbone,
        pretrained_weights=config.model.pretrained_weights if hasattr(
            config.model, 'pretrained_weights') else None,
        freeze_layers=config.model.freeze_layers if hasattr(
            config.model, 'freeze_layers') else "none"
    ).to(device)

    return model
