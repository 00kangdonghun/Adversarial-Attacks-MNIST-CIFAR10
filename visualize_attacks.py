import matplotlib.pyplot as plt
import os
import torch

def visualize_attack_examples(model, loader, attack_fn_dict, device, num_images=5, save_path="results/attack_visualization.png"):
    model.eval()
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    fig, axes = plt.subplots(len(attack_fn_dict) + 1, num_images, figsize=(num_images * 2, 2.5 * (len(attack_fn_dict) + 1)))

    # Clean 이미지 시각화
    with torch.no_grad():
        for i in range(num_images):
            img = images[i].cpu().squeeze()
            pred = model(images[i].unsqueeze(0)).argmax(dim=1).item()
            if img.ndim == 2:
                axes[0, i].imshow(img, cmap="gray")
            else:
                axes[0, i].imshow(img.permute(1, 2, 0))  # (C,H,W) → (H,W,C)
            axes[0, i].set_title(f"Clean\nPred: {pred}")
            axes[0, i].axis("off")

    # 공격 이미지 시각화
    for row_idx, (attack_name, attack_fn) in enumerate(attack_fn_dict.items(), start=1):
        adv_images = attack_fn(model, images[:num_images], labels[:num_images])
        with torch.no_grad():
            for i in range(num_images):
                img = adv_images[i].cpu().squeeze()
                pred = model(adv_images[i].unsqueeze(0)).argmax(dim=1).item()
                if img.ndim == 2:
                    axes[row_idx, i].imshow(img, cmap="gray")
                else:
                    axes[row_idx, i].imshow(img.permute(1, 2, 0))
                axes[row_idx, i].set_title(f"{attack_name}\nPred: {pred}")
                axes[row_idx, i].axis("off")

    plt.tight_layout()

    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 저장 및 출력
    plt.savefig(save_path, dpi=200)
    plt.show()
