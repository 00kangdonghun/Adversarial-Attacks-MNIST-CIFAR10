import torch
import torch.nn.functional as F

def fgsm_targeted(model, x, target, eps):
    x.requires_grad = True
    output = model(x)
    # CrossEntropy loss 계산 (targeted이면 타겟, untargeted이면 정답)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    # FGSM: 입력을 타겟 방향으로 이동 (sign 방향 반대로)
    x_adv = x - eps * x.grad.sign()
    return torch.clamp(x_adv.detach(), 0, 1)

def fgsm_untargeted(model, x, label, eps):
    x.requires_grad = True
    output = model(x)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    # FGSM: 입력을 오답 방향으로 이동 (gradient 방향으로)
    x_adv = x + eps * x.grad.sign()
    return torch.clamp(x_adv.detach(), 0, 1)
