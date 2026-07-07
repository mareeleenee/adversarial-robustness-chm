'''Adversarial training CIFAR10 — CHM v3: TRADES + worst-case-over-hull KL + AWP + EMA.

Recipe (see PLAN_BEAT_CAP.md for the full strategy and references):
  - TRADES loss: CE(clean) + beta * KL(adv || clean), KL-PGD-10 training attack
    (Zhang et al. 2019, arXiv:1901.08573)
  - CHM hull term: the KL regularizer is the *worst case* over N_hull independent
    KL-PGD solutions (hull vertices) plus K_mix Dirichlet convex combinations of
    their deltas (hull interior). With --N_hull 1 --K_mix 0 this is exactly TRADES.
    (hull interior sampling adapted from Zhou et al. 2021, arXiv:2006.11627)
  - AWP adversarial weight perturbation, TRADES-AWP variant
    (Wu et al., NeurIPS 2020, arXiv:2004.05884; github.com/csdongxian/AWP)
  - EMA weight averaging (Gowal et al. 2020, arXiv:2010.03593)
  - model selection by PGD-10 robust acc on a held-out 1k train split
    (robust overfitting: Rice et al. 2020, arXiv:2002.11569)
  - 120 epochs, SGD 0.1, /10 at epochs 80 and 100 — matches the CAP paper's setup
    (Mohajer Hamidi & Ye, ICASSP 2024) for a fair comparison.

Unlike v1/v2, data pipelines here are UNNORMALIZED ([0,1] tensors); normalization
happens inside the forward pass. Checkpoints keep the same {'net','ema'} format
(DataParallel 'module.' prefix), so eval_autoattack.py / eval_checkpoint.py work as-is.
'''
import copy
import datetime
import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import ResNet18
from utils import progress_bar

parser = argparse.ArgumentParser(description='CIFAR10 CHM v3: TRADES-hull + AWP + EMA')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--lr_drops', default=[80, 100], type=int, nargs='+')
parser.add_argument('--steps', default=10, type=int, help='KL-PGD training attack steps')
parser.add_argument('--beta', default=6.0, type=float, help='TRADES robustness weight')
parser.add_argument('--N_hull', default=1, type=int, help='KL-PGD restarts = hull vertices')
parser.add_argument('--K_mix', default=0, type=int, help='Dirichlet interior samples')
parser.add_argument('--awp_gamma', default=5e-3, type=float, help='0 disables AWP')
parser.add_argument('--awp_lr', default=0.01, type=float)
parser.add_argument('--awp_warmup', default=10, type=int, help='epochs before AWP starts')
parser.add_argument('--ema_decay', default=0.995, type=float)
parser.add_argument('--random_erase', default=0.0, type=float,
                    help='RandomErasing prob (phase-2 augmentation lever, default off)')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--max_batches', default=0, type=int, help='debug: limit train batches')
parser.add_argument('--run_name', default=None, type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == 'cifar10':
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD  = (0.2023, 0.1994, 0.2010)
    DATASET_CLS = torchvision.datasets.CIFAR10
    NUM_CLASSES = 10
else:  # cifar100
    CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR_STD  = (0.2673, 0.2564, 0.2762)
    DATASET_CLS = torchvision.datasets.CIFAR100
    NUM_CLASSES = 100
MEAN_T = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
STD_T  = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

eps = 8 / 255
alpha = 2 / 255

print(f"=== Starting Training: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} ===")


def normalize(x):
    return (x - MEAN_T) / STD_T


# Data — NO Normalize in the transform; tensors stay in [0,1]
print('==> Preparing data..')
train_tf = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
if args.random_erase > 0:
    train_tf.append(transforms.RandomErasing(p=args.random_erase))
transform_train = transforms.Compose(train_tf)
transform_test = transforms.Compose([transforms.ToTensor()])

trainset_full = DATASET_CLS(
    root='./data', train=True, download=True, transform=transform_train)
valset_base = DATASET_CLS(
    root='./data', train=True, download=True, transform=transform_test)

# same split as v2 (generator seed 0) so val numbers are comparable across versions
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
testset = DATASET_CLS(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = ResNet18(num_classes=NUM_CLASSES).to(device)
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


def kl_per_sample(logits_adv, logp_clean):
    # log-space KL (log_target=True) is NaN-safe: softmax targets underflow to
    # exact 0 for logit gaps >~88 in fp32, and 0*log(0) poisons the loss
    return F.kl_div(F.log_softmax(logits_adv, dim=1), logp_clean,
                    reduction='none', log_target=True).sum(dim=1)


def trades_attack(model, x, steps=10):
    """KL-PGD attack of TRADES (gaussian 1e-3 init), in [0,1] space."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logp_clean = F.log_softmax(model(normalize(x)), dim=1)
    x_adv = (x + 0.001 * torch.randn_like(x)).clamp(0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = kl_per_sample(model(normalize(x_adv)), logp_clean).sum()
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


def pgd_ce_attack(model, x, y, steps=10, n_eps=eps, n_alpha=alpha):
    """Standard CE-PGD with uniform random start, in [0,1] space (for eval)."""
    was_training = model.training
    model.eval()
    x_adv = (x + torch.empty_like(x).uniform_(-n_eps, n_eps)).clamp(0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = F.cross_entropy(model(normalize(x_adv)), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + n_alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - n_eps), x + n_eps).clamp(0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


# ----- AWP (TRADES-AWP variant, after github.com/csdongxian/AWP) -----
AWP_EPS = 1e-20


def diff_in_weights(model, proxy):
    diff = OrderedDict()
    for (k, w_old), (_, w_new) in zip(model.state_dict().items(),
                                      proxy.state_dict().items()):
        if w_old.dim() > 1 and 'weight' in k:
            d = w_new - w_old
            diff[k] = w_old.norm() / (d.norm() + AWP_EPS) * d
    return diff


@torch.no_grad()
def add_into_weights(model, diff, coeff):
    for name, param in model.named_parameters():
        if name in diff:
            param.add_(coeff * diff[name])


class TradesAWP:
    def __init__(self, model, gamma, lr):
        self.model = model
        self.proxy = copy.deepcopy(model)
        self.optim = optim.SGD(self.proxy.parameters(), lr=lr)
        self.gamma = gamma

    def calc_awp(self, x, x_adv, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        logits_clean = self.proxy(normalize(x))
        loss_nat = F.cross_entropy(logits_clean, targets)
        loss_rob = kl_per_sample(self.proxy(normalize(x_adv)),
                                 F.log_softmax(logits_clean, dim=1)).mean()
        loss = -(loss_nat + beta * loss_rob)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        return diff_in_weights(self.model, self.proxy)

    def perturb(self, diff):
        add_into_weights(self.model, diff, self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, -self.gamma)


awp = TradesAWP(net, args.awp_gamma, args.awp_lr) if args.awp_gamma > 0 else None

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args.lr_drops, gamma=0.1)

run_name = args.run_name or (
    f"v3_res18_b{args.beta}_Nh{args.N_hull}_K{args.K_mix}"
    f"_awp{args.awp_gamma}_ep{args.epochs}_s{args.seed}"
)
os.makedirs('checkpoint', exist_ok=True)

print("=== Run Config ===")
print(f"run_name={run_name}")
print(f"TRADES: beta={args.beta} KL-PGD steps={args.steps} eps={eps:.5f} alpha={alpha:.5f}")
print(f"Hull: N_hull={args.N_hull} K_mix={args.K_mix}")
print(f"AWP: gamma={args.awp_gamma} lr={args.awp_lr} warmup={args.awp_warmup}")
print(f"ema_decay={args.ema_decay} epochs={args.epochs} lr_drops={args.lr_drops} "
      f"random_erase={args.random_erase} seed={args.seed}")
print("==================")


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    sums = {'loss': 0.0, 'nat': 0.0, 'rob': 0.0}
    correct_adv, correct_clean, total = 0, 0, 0
    awp_active = awp is not None and epoch >= args.awp_warmup

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        # hull vertices: independent KL-PGD solutions
        adv_list = [trades_attack(net, inputs, steps=args.steps)
                    for _ in range(args.N_hull)]

        # representative worst vertex for the AWP proxy step
        if args.N_hull == 1:
            x_adv_worst = adv_list[0]
        else:
            with torch.no_grad():
                net.eval()
                logp_clean = F.log_softmax(net(normalize(inputs)), dim=1)
                kls = torch.stack([kl_per_sample(net(normalize(xa)), logp_clean)
                                   for xa in adv_list], dim=0)        # [N, B]
                widx = kls.argmax(dim=0)                              # [B]
                net.train()
            x_stack = torch.stack(adv_list, dim=0)                    # [N, B, ...]
            x_adv_worst = x_stack[widx, torch.arange(inputs.size(0))]

        if awp_active:
            diff = awp.calc_awp(inputs, x_adv_worst, targets, args.beta)
            awp.perturb(diff)

        net.train()
        optimizer.zero_grad(set_to_none=True)

        logits_clean = net(normalize(inputs))
        loss_nat = F.cross_entropy(logits_clean, targets)
        logp_clean = F.log_softmax(logits_clean, dim=1)

        # worst-case KL over hull samples: vertices + Dirichlet interior mixes
        kl_list = [kl_per_sample(net(normalize(xa)), logp_clean) for xa in adv_list]
        if args.K_mix > 0 and args.N_hull >= 2:
            # mixes of feasible deltas are feasible: the Linf ball and [0,1] box are convex
            deltas = torch.stack([xa - inputs for xa in adv_list], dim=0)  # [N,B,...]
            for _ in range(args.K_mix):
                w = torch.distributions.Dirichlet(
                    torch.ones(args.N_hull, device=device)).sample((inputs.size(0),))
                w = w.t().view(args.N_hull, -1, 1, 1, 1)
                x_mix = (inputs + (w * deltas).sum(dim=0)).clamp(0.0, 1.0)
                kl_list.append(kl_per_sample(net(normalize(x_mix)), logp_clean))
        kl_stack = torch.stack(kl_list, dim=0)                        # [N+K, B]
        loss_rob = kl_stack.max(dim=0).values.mean()

        loss = loss_nat + args.beta * loss_rob
        if not torch.isfinite(loss):
            # never let one bad batch poison the weights; loud so it shows in logs
            print(f'\nWARNING: non-finite loss at epoch {epoch} batch {batch_idx}, '
                  f'skipping update (nat={loss_nat.item():.3f} rob={loss_rob.item():.3f})')
            optimizer.zero_grad(set_to_none=True)
            if awp_active:
                awp.restore(diff)
            continue
        loss.backward()
        optimizer.step()

        if awp_active:
            awp.restore(diff)
        ema_update(args.ema_decay)

        sums['loss'] += loss.item()
        sums['nat'] += loss_nat.item()
        sums['rob'] += loss_rob.item()
        with torch.no_grad():
            correct_clean += logits_clean.argmax(1).eq(targets).sum().item()
            correct_adv += net(normalize(x_adv_worst)).argmax(1).eq(targets).sum().item()
        total += targets.size(0)

        progress_bar(
            batch_idx, len(trainloader),
            'L: %.3f | nat: %.3f | rob: %.3f | Clean: %.1f%% | Adv: %.1f%% | awp:%d' % (
                sums['loss'] / (batch_idx + 1),
                sums['nat'] / (batch_idx + 1),
                sums['rob'] / (batch_idx + 1),
                100. * correct_clean / total,
                100. * correct_adv / total,
                int(awp_active)))


@torch.no_grad()
def clean_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        correct += model(normalize(inputs)).argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return 100. * correct / total


def robust_acc(model, loader, steps=10):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        x_adv = pgd_ce_attack(model, inputs, targets, steps=steps)
        with torch.no_grad():
            correct += model(normalize(x_adv)).argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return 100. * correct / total


best_robust = {'net': 0.0, 'ema': 0.0}

for epoch in range(args.epochs):
    train(epoch)

    val_rob_net = robust_acc(net, valloader)
    val_rob_ema = robust_acc(ema_net, valloader)
    val_clean_ema = clean_acc(ema_net, valloader)
    print(f"[Val] epoch={epoch} rob(net)={val_rob_net:.2f}% rob(ema)={val_rob_ema:.2f}% "
          f"clean(ema)={val_clean_ema:.2f}%")

    for tag, rob in (('net', val_rob_net), ('ema', val_rob_ema)):
        if rob > best_robust[tag]:
            best_robust[tag] = rob
            torch.save({'net': net.state_dict(), 'ema': ema_net.state_dict(),
                        'val_robust': rob, 'epoch': epoch},
                       f'./checkpoint/{run_name}_best_{tag}.pth')
            print(f'Saved best_{tag} (val robust {rob:.2f}%)')

    if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
        for tag, model in (('net', net), ('ema', ema_net)):
            tc = clean_acc(model, testloader)
            tr = robust_acc(model, testloader, steps=20)
            print(f"[Test/{tag}] epoch={epoch} clean={tc:.2f}% PGD-20={tr:.2f}%")

    torch.save({'net': net.state_dict(), 'ema': ema_net.state_dict(), 'epoch': epoch},
               f'./checkpoint/{run_name}_last.pth')
    scheduler.step()

print(f"Best val robust: net={best_robust['net']:.2f}% ema={best_robust['ema']:.2f}%")
print(f"=== Finished Training: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} ===")
