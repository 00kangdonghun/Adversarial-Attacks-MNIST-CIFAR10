import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from models.mnist_model import SimpleMNISTModel
from utils.data_loader import get_mnist_loaders
from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and data
model = SimpleMNISTModel().to(device)
train_loader, test_loader = get_mnist_loaders()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train (3 epoch)
print("Training model...")
model.train()
for epoch in range(3):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Evaluate
def evaluate(model, loader, attack_fn=None, name="Clean"):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack_fn:
            x = attack_fn(model, x, y)
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    print(f"{name} Accuracy: {correct / total * 100:.2f}%")

# Attack parameters
eps = 0.3
eps_step = 0.01
k = 10

# Run evaluations
# 원본 이미지
evaluate(model, test_loader, None, "Clean")
evaluate(model, test_loader, lambda m, x, y: fgsm_untargeted(m, x, y, eps), "FGSM Untargeted")
evaluate(model, test_loader, lambda m, x, y: fgsm_targeted(m, x, (y + 1) % 10, eps), "FGSM Targeted")
evaluate(model, test_loader, lambda m, x, y: pgd_untargeted(m, x, y, eps, eps_step, k), "PGD Untargeted")
evaluate(model, test_loader, lambda m, x, y: pgd_targeted(m, x, (y + 1) % 10, eps, eps_step, k), "PGD Targeted")

# Visualization
def visualize(model, loader, attack_dict, save_path, num_images=5):
    model.eval()
    # 테스트셋에서 이미지 일부 추출
    x, y = next(iter(loader))
    x, y = x[:num_images].to(device), y[:num_images].to(device)

    fig, axes = plt.subplots(len(attack_dict) + 1, num_images, figsize=(num_images * 2, 3 * (len(attack_dict)+1)))

    # 원본 이미지 출력
    with torch.no_grad():
        for i in range(num_images):
            img = x[i].cpu().squeeze()
            pred = model(x[i].unsqueeze(0)).argmax(1).item()
            axes[0, i].imshow(img, cmap="gray")
            axes[0, i].set_title(f"Clean\nPred: {pred}")
            axes[0, i].axis("off")
    # 각 공격에 대해 이미지 생성 및 출력
    for row_idx, (name, attack_fn) in enumerate(attack_dict.items(), 1):
        x_adv = attack_fn(model, x.clone(), y.clone())
        with torch.no_grad():
            for i in range(num_images):
                img = x_adv[i].cpu().squeeze()
                pred = model(x_adv[i].unsqueeze(0)).argmax(1).item()
                axes[row_idx, i].imshow(img, cmap="gray")
                axes[row_idx, i].set_title(f"{name}\nPred: {pred}")
                axes[row_idx, i].axis("off")
    
    # 결과 이미지 저장 및 출력
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 시각화 실행
visualize(model, test_loader, {
    "FGSM Untargeted": lambda m, x, y: fgsm_untargeted(m, x, y, eps),
    "PGD Untargeted": lambda m, x, y: pgd_untargeted(m, x, y, eps, eps_step, k),
    "FGSM Targeted": lambda m, x, y: fgsm_targeted(m, x, (y + 1) % 10, eps),
    "PGD Targeted": lambda m, x, y: pgd_targeted(m, x, (y + 1) % 10, eps, eps_step, k)
}, "results/mnist_attacks.png")
