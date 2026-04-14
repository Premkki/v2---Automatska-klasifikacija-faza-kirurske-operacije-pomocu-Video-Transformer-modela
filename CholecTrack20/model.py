import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

NUM_CLASSES = 7

def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Učitava Video Swin Transformer (Small) s pretrained težinama
    i zamjenjuje klasifikacijski sloj za naš broj klasa.
    """
    if pretrained:
        weights = Swin3D_T_Weights.DEFAULT
        model = swin3d_t(weights=weights)
        print("[INFO] Učitane pretrained težine (Kinetics-400)")
    else:
        model = swin3d_t(weights=None)
        print("[INFO] Model inicijaliziran bez pretrained težina")

    # Zamijeni klasifikacijski sloj
    # Original: 400 klasa (Kinetics-400)
    # Naš zadatak: 7 klasa (kirurške faze)
    in_features = model.head.in_features
    #model.head = nn.Linear(in_features, num_classes)
    #print(f"[INFO] Klasifikacijski sloj zamijenjen: {in_features} -> {num_classes} klasa")

    return model, in_features


if __name__ == "__main__":
    # Provjera modela
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Koristi se: {device}")

    model, in_features = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model.head = nn.Linear(in_features, NUM_CLASSES)

    model = model.to(device)

    # Test s jednim lažnim batchem
    # Ulaz: (B, C, T, H, W) = (2, 3, 16, 224, 224)
    dummy_input = torch.randn(2, 3, 16, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n[INFO] Ulazni oblik:  {dummy_input.shape}")
    print(f"[INFO] Izlazni oblik: {output.shape}")
    print(f"[INFO] Očekivano:     torch.Size([2, 7])")

    # Provjeri broj parametara
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Ukupno parametara:    {total_params:,}")
    print(f"[INFO] Trainable parametara: {trainable_params:,}")