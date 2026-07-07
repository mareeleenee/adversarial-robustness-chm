'''Adversarial training CIFAR10 — CHM v2.

Improvements over main_adv_CHM_baseline.py:
  - worst-case CE over the N_hull adversarial variants, i.e. PGD with random
    restarts (PGD-AT: Madry et al. 2018, arXiv:1706.06083)
  - hull interior sampling: margin enforced on Dirichlet convex combinations
    of the adversarial deltas, not just the vertices (--hull_mode mix/both);
    Dirichlet sampling in a convex hull of perturbations adapted from NLP
    defenses (Zhou et al. 2021, arXiv:2006.11627; Dong et al., ICLR 2021
    "Towards Robustness Against Natural Language Word Substitutions")
  - EMA weight averaging (weight averaging for robustness: Izmailov et al.
    2018, arXiv:1803.05407; EMA in adversarial training: Gowal et al. 2020,
    arXiv:2010.03593)
  - model selection by PGD robust accuracy on a held-out train split, not by
    clean test accuracy (robust overfitting / early stopping: Rice et al.
    2020, arXiv:2002.11569)
  - PGD-10 training attack by default, no gradient clipping by default
With --N_hull 1 --lam_hull 0 this is exactly standard PGD-AT.
'''
import copy
import datetime
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torchattacks

from models import ResNet18
from utils import progress_bar

parser = argparse.ArgumentParser(description='CIFAR10 CHM v2 adversarial training')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--steps', default=10, type=int, help='PGD training steps')
parser.add_argument('--N_hull', default=2, type=int, help='adversarial variants per image')
parser.add_argument('--lam_hull', default=0.005, type=float, help='hull margin weight')
parser.add_argument('--hull_mode', default='both', choices=['vertex', 'mix', 'both'],
                    help='margin on worst vertex, on convex-combination samples, or both')
parser.add_argument('--K_mix', default=2, type=int, help='convex-combination samples per image')
parser.add_argument('--ema_decay', default=0.999, type=float)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--ramp_epochs', default=5, type=int)
parser.add_argument('--clip_grad', default=0.0, type=float, help='0 disables clipping')
parser.add_argument('--max_batches', default=0, type=int, help='debug: limit train batches per epoch')
parser.add_argument('--run_name', default=None, type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

eps = 8 / 255
alpha = 2 / 255

print(f"=== Starting Training: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} ===")

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

trainset_full = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
# held-out (unaugmented) split from the train set for robust model selection
valset_base = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_test)

g = torch.Generator().manual_seed(0)
perm = torch.randperm(50000, generator=g)
val_idx = perm[:1000].tolist()
train_idx = perm[1000:].tolist()

trainset = torch.utils.data.Subset(trainset_full, train_idx)
valset = torch.utils.data.Subset(valset_base, val_idx)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=200, shuffle=False, num_workers=2)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = ResNet18().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

ema_net = copy.deepcopy(net)
for p in ema_net.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def ema_update(decay):
    ema_params = dict(ema_net.named_parameters())
    for name, p in net.named_parameters():
        ema_params[name].mul_(decay).add_(p, alpha=1 - decay)
    ema_buffers = dict(ema_net.named_buffers())
    for name, b in net.named_buffers():
        ema_buffers[name].copy_(b)


atk = torchattacks.PGD(net, eps=eps, alpha=alpha, steps=args.steps, random_start=True)
atk.set_normalization_used(mean=CIFAR_MEAN, std=CIFAR_STD)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

run_name = args.run_name or (
    f"v2_res18_steps{args.steps}_Nh{args.N_hull}_lam{args.lam_hull}"
    f"_{args.hull_mode}_ep{args.epochs}"
)
os.makedirs('checkpoint', exist_ok=True)

print("=== Run Config ===")
print(f"run_name={run_name}")
print(f"Train PGD: eps={eps:.5f} alpha={alpha:.5f} steps={args.steps}")
print(f"N_hull={args.N_hull} lam_hull={args.lam_hull} hull_mode={args.hull_mode} K_mix={args.K_mix}")
print(f"ema_decay={args.ema_decay} warmup={args.warmup_epochs} ramp={args.ramp_epochs}")
print("==================")


def margin_loss(logits_stack, targets):
    """softplus(max_other - correct) on the worst variant per example.
    logits_stack: [N, B, C]"""
    logits_stack = torch.clamp(logits_stack, min=-30.0, max=30.0)
    N, B, C = logits_stack.shape
    t = targets.view(1, B, 1).expand(N, B, 1)
    y_logit = logits_stack.gather(2, t).squeeze(2)
    tmp = logits_stack.clone()
    tmp.scatter_(2, t, -1e9)
    max_other = tmp.max(dim=2).values
    viol = F.softplus(max_other - y_logit)          # [N, B]
    return viol.max(dim=0).values.mean()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    sums = {'loss': 0.0, 'ce': 0.0, 'hull': 0.0}
    correct, total = 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        # adversarial variants (attack in eval mode)
        net.eval()
        adv_list = [atk(inputs, targets).detach() for _ in range(args.N_hull)]
        optimizer.zero_grad(set_to_none=True)
        net.train()

        logits_list = [net(x_adv) for x_adv in adv_list]
        logits_stack = torch.stack(logits_list, dim=0)  # [N, B, C]

        # worst-case CE over variants (PGD with restarts)
        ce_stack = torch.stack(
            [F.cross_entropy(lg, targets, reduction='none') for lg in logits_list], dim=0)
        worst_idx = ce_stack.argmax(dim=0)              # [B]
        loss_ce = ce_stack.max(dim=0).values.mean()

        # hull margin term (ramp reaches 1/ramp_epochs on the first post-warmup epoch,
        # 1.0 at warmup+ramp; the old formula left the first post-warmup epoch at 0)
        ramp = min(1.0, max(0.0, (epoch - args.warmup_epochs + 1) / max(1, args.ramp_epochs)))
        eff_lam = args.lam_hull * (0.0 if epoch < args.warmup_epochs else ramp)

        loss_hull = torch.tensor(0.0, device=device)
        if eff_lam > 0:
            margin_stack = []
            if args.hull_mode in ('vertex', 'both'):
                margin_stack.append(logits_stack)
            if args.hull_mode in ('mix', 'both') and args.N_hull >= 2:
                # Dirichlet convex combinations of the deltas: interior of the hull.
                # The Linf ball and [0,1] box are convex, so mixes are valid perturbations.
                deltas = torch.stack([x_adv - inputs for x_adv in adv_list], dim=0)  # [N,B,...]
                mix_logits = []
                for _ in range(args.K_mix):
                    w = torch.distributions.Dirichlet(
                        torch.ones(args.N_hull, device=device)).sample((inputs.size(0),))  # [B,N]
                    w = w.t().view(args.N_hull, -1, 1, 1, 1)
                    x_mix = inputs + (w * deltas).sum(dim=0)
                    mix_logits.append(net(x_mix))
                margin_stack.append(torch.stack(mix_logits, dim=0))
            loss_hull = margin_loss(torch.cat(margin_stack, dim=0), targets)

        loss = loss_ce + eff_lam * loss_hull
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_grad)
        optimizer.step()
        ema_update(args.ema_decay)

        sums['loss'] += loss.item()
        sums['ce'] += loss_ce.item()
        sums['hull'] += loss_hull.item()
        outputs_worst = logits_stack[worst_idx, torch.arange(targets.size(0))]
        _, predicted = outputs_worst.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(trainloader),
            'L: %.3f | CE*: %.3f | Lh: %.3f | lam*: %.4f | AdvAcc: %.3f%% (%d/%d)' % (
                sums['loss'] / (batch_idx + 1),
                sums['ce'] / (batch_idx + 1),
                sums['hull'] / (batch_idx + 1),
                eff_lam,
                100. * correct / total, correct, total))


@torch.no_grad()
def clean_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        correct += model(inputs).argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return 100. * correct / total


def robust_acc(model, loader, steps=10):
    model.eval()
    atk_eval = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    atk_eval.set_normalization_used(mean=CIFAR_MEAN, std=CIFAR_STD)
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_adv = atk_eval(inputs, targets)
        with torch.no_grad():
            correct += model(inputs_adv).argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return 100. * correct / total


best_robust = {'net': 0.0, 'ema': 0.0}

for epoch in range(args.epochs):
    train(epoch)

    # model selection on held-out train split (PGD-10)
    val_rob_net = robust_acc(net, valloader)
    val_rob_ema = robust_acc(ema_net, valloader)
    val_clean_ema = clean_acc(ema_net, valloader)
    print(f"[Val] epoch={epoch} rob(net)={val_rob_net:.2f}% rob(ema)={val_rob_ema:.2f}% "
          f"clean(ema)={val_clean_ema:.2f}%")

    for tag, model, rob in (('net', net, val_rob_net), ('ema', ema_net, val_rob_ema)):
        if rob > best_robust[tag]:
            best_robust[tag] = rob
            torch.save({'net': net.state_dict(), 'ema': ema_net.state_dict(),
                        'val_robust': rob, 'epoch': epoch},
                       f'./checkpoint/{run_name}_best_{tag}.pth')
            print(f'Saved best_{tag} (val robust {rob:.2f}%)')

    # periodic full test-set eval (clean + PGD-20)
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        test_clean = clean_acc(ema_net, testloader)
        test_rob = robust_acc(ema_net, testloader, steps=20)
        print(f"[Test/EMA] epoch={epoch} clean={test_clean:.2f}% PGD-20={test_rob:.2f}%")

    torch.save({'net': net.state_dict(), 'ema': ema_net.state_dict(), 'epoch': epoch},
               f'./checkpoint/{run_name}_last.pth')
    scheduler.step()

print(f"Best val robust: net={best_robust['net']:.2f}% ema={best_robust['ema']:.2f}%")
print(f"=== Finished Training: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} ===")
