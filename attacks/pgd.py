import torch
import torch.nn.functional as F

def pgd_targeted(model, x, target, eps, eps_step, k):
    x_adv = x.clone().detach()
    for _ in range(k):
        x_adv.requires_grad = True
        output = model(x_adv)
        # CrossEntropy loss 계산 (targeted이면 타겟, untargeted이면 정답)
        loss = F.cross_entropy(output, target) 
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign()
        # PGD: 입력을 타겟 방향으로 이동 (sign 방향 반대로)
        x_adv = x_adv - eps_step * grad 
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
    return x_adv

def pgd_untargeted(model, x, label, eps, eps_step, k):
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = F.cross_entropy(output, label)  
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign()
        x_adv = x_adv + eps_step * grad  
        # PGD: 입력을 오답 방향으로 이동 (gradient 방향으로)
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()

    return x_adv



















# def pgd_attack(model, x, y, eps, eps_step, k, targeted=False):
#     x_adv = x.clone().detach()
#     for _ in range(k):
#         x_adv.requires_grad = True
#         output = model(x_adv)
#         # CrossEntropy loss 계산 (targeted이면 타겟, untargeted이면 정답)
#         loss = F.cross_entropy(output, y)
#         model.zero_grad()
#         loss.backward()

#         grad = x_adv.grad.sign()
#         # 타겟 공격이면 grad 방향 반대로 (목표 class로 가도록)
#         if targeted:
#             x_adv = x_adv - eps_step * grad
#         else:
#             x_adv = x_adv + eps_step * grad

#         # Perturbation을 원본 주변 eps-ball 내로 제한
#         x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
#         # 픽셀 값을 [0, 1] 범위로 클리핑 (이미지 유효 범위)
#         x_adv = torch.clamp(x_adv, 0, 1)
#         x_adv = x_adv.detach()
#     return x_adv
