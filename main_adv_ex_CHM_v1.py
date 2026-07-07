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

import torchattacks


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=20, type=int)
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

steps = 3  # start with 10 for speed, later 20
N_hull = 2      # start small (2). later try 4, 8
lam_hull = 0.01  # start small. later tune 0.1, 0.3, 0.

hull_warmup_epochs = 5
hull_ramp_epochs = 5

atk = torchattacks.PGD(net, eps=eps, alpha=alpha, steps=steps, random_start=True)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
num_epochs = args.epochs
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

    viol = max_other - y_logit                      # [N,B]

    # smoother than ReLU, less brittle gradients
    return F.softplus(viol).mean()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_loss_clean = 0
    train_loss_hull = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = criterion(outputs, targets)

        # Clean classification loss (helps preserve clean accuracy)
        outputs_clean = net(inputs)
        loss_clean = criterion(outputs_clean, targets)

        # Generate multiple adversarial samples (different random starts)
        net.eval()
        logits_list = []
        for _ in range(N_hull):
            inputs_adv = atk(inputs, targets)
            logits_adv = net(inputs_adv)
            logits_list.append(logits_adv)
        net.train()

        logits_stack = torch.stack(logits_list, dim=0)

        if epoch < hull_warmup_epochs:
            # warmup phase → only clean loss
            loss_hull = torch.tensor(0.0, device=device)
            loss = loss_clean
        else:
            # after warmup → apply hull regularization
            # loss_hull = hull_margin_loss(logits_stack, targets)
            # loss = loss_clean + lam_hull * loss_hull
            loss_hull = hull_margin_loss(logits_stack, targets)
            ramp = min(1.0, max(0.0, (epoch - hull_warmup_epochs) / hull_ramp_epochs))
            loss = loss_clean + (lam_hull * ramp) * loss_hull

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_loss_clean += loss_clean.item()
        train_loss_hull += loss_hull.item()
        _, predicted = outputs_clean.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            'L: %.3f | Lc: %.3f | Lh: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss/(batch_idx+1),
                train_loss_clean/(batch_idx+1),
                train_loss_hull/(batch_idx+1),
                100.*correct/total,
                correct,
                total
            )
        )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
