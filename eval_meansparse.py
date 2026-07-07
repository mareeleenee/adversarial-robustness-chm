'''MeanSparse post-processing of a trained RN18 checkpoint (arXiv:2406.05927).

Pipeline: load frozen checkpoint into ResNet18MS -> calibrate per-channel mu/sigma
on training data -> select alpha by PGD-10 robust accuracy on the SAME held-out 1k
train split used for model selection everywhere in this repo (never the test set)
-> report test clean / PGD-20 at the chosen alpha (and alpha=0 control).
AutoAttack on the sparsified model is run separately via eval_autoattack_ms.py.
'''
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models.resnet_ms import ResNet18MS

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--key', type=str, default='ema')
parser.add_argument('--alphas', type=float, nargs='+',
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
parser.add_argument('--calib_batches', type=int, default=100)
args = parser.parse_args()

device = 'cuda'
eps, alpha_step = 8 / 255, 2 / 255
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device=device).view(1, 3, 1, 1)
STD = torch.tensor((0.2023, 0.1994, 0.2010), device=device).view(1, 3, 1, 1)


def normalize(x):
    return (x - MEAN) / STD


tt = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tt)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tt)

g = torch.Generator().manual_seed(0)
perm = torch.randperm(50000, generator=g)
val_idx = perm[:1000].tolist()          # same held-out split as training scripts
calib_idx = perm[1000:].tolist()        # calibrate on the training portion only

valloader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(trainset, val_idx), batch_size=200, shuffle=False, num_workers=2)
calibloader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(trainset, calib_idx), batch_size=256, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

net = ResNet18MS().to(device)
state = torch.load(args.ckpt, map_location=device, weights_only=True)[args.key]
state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
missing, unexpected = net.load_state_dict(state, strict=False)
assert not unexpected, f'unexpected keys: {unexpected[:5]}'
assert all('ms' in k for k in missing), f'missing non-MS keys: {missing[:5]}'
net.eval()

# --- calibrate mu/sigma with the checkpoint frozen ---
net.set_calibrating(True)
with torch.no_grad():
    for i, (x, _) in enumerate(calibloader):
        if i >= args.calib_batches:
            break
        net(normalize(x.to(device)))
net.set_calibrating(False)
print(f'calibrated on {args.calib_batches} batches')


def pgd(x, y, steps, loss_fn=F.cross_entropy):
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = loss_fn(net(normalize(x_adv)), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha_step * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
    return x_adv.detach()


def acc(loader, attack_steps=0, max_batches=10**9):
    correct, total = 0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        x_eval = pgd(x, y, attack_steps) if attack_steps else x
        with torch.no_grad():
            correct += net(normalize(x_eval)).argmax(1).eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# --- alpha selection on the held-out val split (PGD-10) ---
print('alpha sweep on val split (PGD-10 robust / clean):')
best_alpha, best_rob = 0.0, -1.0
for a in args.alphas:
    net.set_alpha(a)
    rob = acc(valloader, attack_steps=10)
    cln = acc(valloader)
    print(f'  alpha={a:<5} val_rob={rob:.2f}%  val_clean={cln:.2f}%')
    if rob > best_rob:
        best_rob, best_alpha = rob, a
print(f'chosen alpha={best_alpha} (val rob {best_rob:.2f}%)')

# --- final test evaluation at alpha=0 (control) and chosen alpha ---
for a in sorted({0.0, best_alpha}):
    net.set_alpha(a)
    print(f'[test] alpha={a}: clean={acc(testloader):.2f}%  '
          f'PGD-20={acc(testloader, attack_steps=20):.2f}%')

torch.save({'ms_state': net.state_dict(), 'alpha': best_alpha},
           args.ckpt.replace('.pth', f'_ms.pth'))
print('saved calibrated MS checkpoint')
