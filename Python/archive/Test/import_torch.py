import torch

# 1) Zeige an, ob CUDA verfügbar ist.
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# 2) Falls verfügbar, gib zusätzliche Infos aus:
if cuda_available:
    print(f"Anzahl GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 3) Führe einen kleinen Test durch: 
#    - Erstelle einen Tensor auf der GPU, falls verfügbar.
device = torch.device("cuda" if cuda_available else "cpu")
x = torch.rand((1000, 1000), device=device)
y = torch.rand((1000, 1000), device=device)

# 4) Mache eine Matrixmultiplikation und gib das Ergebnis (oder einen Ausschnitt) aus.
z = torch.matmul(x, y)
print(f"Ergebnis-Tensor z (shape={z.shape}) liegt auf Gerät: {z.device}")

# 5) Optional: Drucke einen kleinen Teil des Tensors (z.B. Mittelwert)
print(f"Mean of z: {z.mean().item()}")
