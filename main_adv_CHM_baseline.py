'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import datetime

import torchattacks

datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"=== Starting Training: {datetime_str} ===")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=20, type=int)
# CHM params
parser.add_argument('--lam_hull', default=0, type=float, help='hull regularization strength')
parser.add_argument('--N_hull', default=2, type=int, help='number of hull adversarial variants')
# AT strength
parser.add_argument('--steps', default=5, type=int, help='PGD training steps')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

eps = 8/255
alpha = 2/255

# Evaluation attack
eps_eval = 8/255
alpha_eval = 2/255
steps_eval = 20

steps = args.steps
N_hull = args.N_hull
lam_hull = args.lam_hull


hull_warmup_epochs = 5
hull_ramp_epochs = 5

# --- constants for normalization (use the same as your transforms) ---
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

atk = torchattacks.PGD(net, eps=eps, alpha=alpha, steps=steps, random_start=True)

atk.set_normalization_used(mean=CIFAR_MEAN, std=CIFAR_STD)

print("=== Run Config ===")
print(f"Model: ResNet18")
print(f"Train PGD: eps={eps}, alpha={alpha}, steps={steps}")
print(f"Eval PGD: eps={eps_eval}, alpha={alpha_eval}, steps={steps_eval}")
print(f"N_hull={N_hull}, lam_hull={lam_hull}")
print(f"Warmup={hull_warmup_epochs}, Ramp={hull_ramp_epochs}")
print("==================")

num_epochs = args.epochs
run_name = f"res18_steps{steps}_Nh{N_hull}_lam{lam_hull}_ep{num_epochs}"
# if args.run_name is None:
#     run_name = f"res18_steps{steps}_Nh{N_hull}_lam{lam_hull}_ep{num_epochs}"
# else:
#     run_name = args.run_name
ckpt_path = f'./checkpoint/{run_name}.pth'

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.isfile(ckpt_path), f'Error: checkpoint not found: {ckpt_path}'

    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

    print(f'Resumed from epoch {start_epoch}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def hull_margin_loss(logits_stack, targets):
    """
    logits_stack: [N, B, C]
    targets: [B]
    """
    # Optional safety clamp to prevent insane logits from dominating early training
    logits_stack = torch.clamp(logits_stack, min=-30.0, max=30.0)

    N, B, C = logits_stack.shape
    t = targets.view(1, B, 1).expand(N, B, 1)

    y_logit = logits_stack.gather(2, t).squeeze(2)  # [N,B]

    tmp = logits_stack.clone()
    tmp.scatter_(2, t, -1e9)
    max_other = tmp.max(dim=2).values               # [N,B]

    # viol = max_other - y_logit                      # [N,B]

    # # smoother than ReLU, less brittle gradients
    # return F.softplus(viol).mean()
    viol = max_other - y_logit          # [N, B]
    loss_per_point = F.softplus(viol)   # [N, B]

    # take worst hull sample per image
    worst_per_example = loss_per_point.max(dim=0).values  # [B]

    return worst_per_example.mean()



def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0.0
    train_loss_adv = 0.0
    train_loss_hull = 0.0

    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # --- Generate N_hull adversarial variants (random starts) ---
        # Attack generation is more stable in eval mode
       # --- Generate N_hull adversarial variants (in eval mode) ---
        net.eval()
        adv_list = []
        for _ in range(N_hull):
            x_adv = atk(inputs, targets)
            adv_list.append(x_adv.detach())

        # IMPORTANT: clear any grads possibly created during attack generation
        optimizer.zero_grad(set_to_none=True)

        # --- Compute all training losses in train mode ---
        net.train()

        # logits for hull term (train mode!)
        logits_list = [net(x_adv) for x_adv in adv_list]
        logits_stack = torch.stack(logits_list, dim=0)  # [N_hull, B, C]

        # adversarial CE (train mode)
        outputs_adv = logits_list[0]   # reuse instead of extra forward
        loss_adv = criterion(outputs_adv, targets)

        # hull regularization
        if epoch < hull_warmup_epochs:
            loss_hull = torch.tensor(0.0, device=device)
            ramp = 0.0
            loss = loss_adv
        else:
            loss_hull = hull_margin_loss(logits_stack, targets)
            ramp = min(1.0, max(0.0, (epoch - hull_warmup_epochs) / hull_ramp_epochs))
            loss = loss_adv + (lam_hull * ramp) * loss_hull

        effective_lam = lam_hull * ramp
        if batch_idx % 100 == 0:  # or == 0 for just first batch
            print(
                f"[dbg] epoch={epoch} batch={batch_idx} "
                f"loss_adv={loss_adv.item():.4f} "
                f"loss_hull={loss_hull.item():.4f} "
                f"lam*={effective_lam:.4f} "
                f"lam*hull={(effective_lam * loss_hull.item()):.4f}"
            )
        
        # right after computing loss_adv, loss_hull, ramp (before loss.backward())

        if batch_idx == 0 and epoch >= hull_warmup_epochs:
            eff = lam_hull * ramp

            optimizer.zero_grad(set_to_none=True)
            loss_adv.backward(retain_graph=True)
            gn_adv = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()

            optimizer.zero_grad(set_to_none=True)
            (eff * loss_hull).backward(retain_graph=True)
            gn_hull = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()

            optimizer.zero_grad(set_to_none=True)
            print(f"[grad] epoch={epoch} gn_adv={gn_adv:.2f} gn_hull={gn_hull:.2f} ratio={gn_hull/(gn_adv+1e-12):.2f}")

        # --- Backprop ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Stats ---
        train_loss += loss.item()
        train_loss_adv += loss_adv.item()
        train_loss_hull += loss_hull.item()

        _, predicted = outputs_adv.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        effective_lam = lam_hull * ramp

        progress_bar(
            batch_idx,
            len(trainloader),
            'L: %.3f | Ladv: %.3f | Lh: %.3f | lam*: %.4f | AdvAcc: %.3f%% (%d/%d)' % (
                train_loss/(batch_idx+1),
                train_loss_adv/(batch_idx+1),
                train_loss_hull/(batch_idx+1),
                effective_lam,
                100.*correct/total,
                correct,
                total
            )
        )


def test(epoch):
    global best_acc
    net.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(
                batch_idx, len(testloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss/(batch_idx+1),
                    100.*correct/total,
                    correct,
                    total
                )
            )
        
    acc = 100.*correct/total
    print(f"\n[Clean Test] Epoch {epoch}: Acc={acc:.3f}%  Loss={test_loss/len(testloader):.3f}\n")

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt_path)
        best_acc = acc


def test_pgd(epoch):
    net.eval()

    robust_loss = 0.0
    correct = 0
    total = 0

    # Create attacker for evaluation (PGD-20)
    atk_eval = torchattacks.PGD(
        net,
        eps=eps_eval,
        alpha=alpha_eval,
        steps=steps_eval,
        random_start=True
    )
    # IMPORTANT: tell attacker about normalization (same as transforms)
    atk_eval.set_normalization_used(mean=CIFAR_MEAN, std=CIFAR_STD)

    # Do NOT wrap the whole loop in no_grad() because attack needs gradients
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate adversarial examples
        inputs_adv = atk_eval(inputs, targets)

        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs_adv = net(inputs_adv)
            loss = criterion(outputs_adv, targets)

        robust_loss += loss.item()
        _, predicted = outputs_adv.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(testloader),
            'PGD-%d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                steps_eval,
                robust_loss/(batch_idx+1),
                100.*correct/total,
                correct,
                total
            )
        )

    robust_acc = 100.*correct/total
    print('PGD-%d Robust Acc: %.3f%%' % (steps_eval, robust_acc))
    return robust_acc


for epoch in range(start_epoch, num_epochs):
    train(epoch)
    test(epoch)

    # Run robust eval every 5 epochs (saves time)
    if (epoch + 1) % 5 == 0:
        print('\n==> PGD evaluation on test set..')
        test_pgd(epoch)

    scheduler.step()


date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"=== Finished Training: {date_str} ===")