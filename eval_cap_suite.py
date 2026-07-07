'''CAP-style attack battery: Clean / FGSM / PGD-20 / PGD-100 / C&W-inf, eps=8/255.

Matches the evaluation columns of CAP (Mohajer Hamidi & Ye, ICASSP 2024,
arXiv:2401.07991) so our checkpoints can fill a CAP Table-1-style row.
Attacks implemented directly in [0,1] image space (normalization inside forward):
  - FGSM: single step, step size = eps (Goodfellow et al. 2015)
  - PGD-k: alpha=2/255, uniform random start (Madry et al. 2018)
  - C&W-inf: PGD-100 maximizing the CW margin loss max(z_other) - z_y
    (Carlini & Wagner loss under Linf, as used in the TRADES/MART/CAP tables)
AutoAttack is covered separately by eval_autoattack.py.
'''
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import ResNet18

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--key', type=str, default='net', help='net or ema')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--batch', default=200, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eps = 8 / 255
alpha = 2 / 255

if args.dataset == 'cifar10':
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    DATASET_CLS, NUM_CLASSES = torchvision.datasets.CIFAR10, 10
else:
    MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    DATASET_CLS, NUM_CLASSES = torchvision.datasets.CIFAR100, 100

MEAN_T = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
STD_T = torch.tensor(STD, device=device).view(1, 3, 1, 1)


def normalize(x):
    return (x - MEAN_T) / STD_T


testset = DATASET_CLS(root='./data', train=False, download=True,
                      transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch, shuffle=False, num_workers=2)

net = ResNet18(num_classes=NUM_CLASSES).to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
net.load_state_dict(ckpt[args.key])
net.eval()


def cw_loss(logits, y):
    """CW margin: max_{j != y} z_j - z_y (to be maximized)."""
    onehot = F.one_hot(y, logits.size(1)).bool()
    z_y = logits[onehot]
    z_other = logits.masked_fill(onehot, -1e9).max(dim=1).values
    return (z_other - z_y).sum()


def attack(x, y, steps, step_size, loss_fn, random_start=True):
    x_adv = x.clone()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = loss_fn(net(normalize(x_adv)), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
    return x_adv.detach()


def ce_loss(logits, y):
    return F.cross_entropy(logits, y, reduction='sum')


ATTACKS = {
    'Clean':   None,
    'FGSM':    dict(steps=1, step_size=eps, loss_fn=ce_loss, random_start=False),
    'PGD-20':  dict(steps=20, step_size=alpha, loss_fn=ce_loss),
    'PGD-100': dict(steps=100, step_size=alpha, loss_fn=ce_loss),
    'CW-inf':  dict(steps=100, step_size=alpha, loss_fn=cw_loss),
}

print(f"ckpt={args.ckpt} key={args.key} dataset={args.dataset} eps=8/255 alpha=2/255")
for name, cfg in ATTACKS.items():
    correct, total = 0, 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        x_eval = x if cfg is None else attack(x, y, **cfg)
        with torch.no_grad():
            correct += net(normalize(x_eval)).argmax(1).eq(y).sum().item()
        total += y.size(0)
    print(f"{name}: {100.0 * correct / total:.2f}%", flush=True)
