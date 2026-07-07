import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import ResNet18
import torchattacks

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint .pth')
parser.add_argument('--pgd_steps', type=int, default=50, help='PGD eval steps')
parser.add_argument('--key', type=str, default='net', help='state dict key in checkpoint (net or ema)')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == 'cifar10':
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD  = (0.2023, 0.1994, 0.2010)
    DATASET_CLS = torchvision.datasets.CIFAR10
    NUM_CLASSES = 10
else:
    CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR_STD  = (0.2673, 0.2564, 0.2762)
    DATASET_CLS = torchvision.datasets.CIFAR100
    NUM_CLASSES = 100

# Data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

testset = DATASET_CLS(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# Model
net = ResNet18(num_classes=NUM_CLASSES).to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint
checkpoint = torch.load(args.ckpt, map_location=device)
net.load_state_dict(checkpoint[args.key])
net.eval()

criterion = nn.CrossEntropyLoss()

# ------------------------
# Clean evaluation
# ------------------------
clean_loss = 0.0
clean_correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        clean_loss += loss.item()
        clean_correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

clean_acc = 100.0 * clean_correct / total
print(f'Clean Acc: {clean_acc:.2f}%')

# ------------------------
# PGD evaluation
# ------------------------
eps_eval = 8/255
alpha_eval = 2/255
steps_eval = args.pgd_steps

atk_eval = torchattacks.PGD(
    net,
    eps=eps_eval,
    alpha=alpha_eval,
    steps=steps_eval,
    random_start=True
)
atk_eval.set_normalization_used(mean=CIFAR_MEAN, std=CIFAR_STD)

robust_loss = 0.0
robust_correct = 0
total = 0

for inputs, targets in testloader:
    inputs, targets = inputs.to(device), targets.to(device)

    inputs_adv = atk_eval(inputs, targets)

    with torch.no_grad():
        outputs_adv = net(inputs_adv)
        loss = criterion(outputs_adv, targets)

    robust_loss += loss.item()
    robust_correct += outputs_adv.argmax(1).eq(targets).sum().item()
    total += targets.size(0)

robust_acc = 100.0 * robust_correct / total
print(f'PGD-{steps_eval} Acc: {robust_acc:.2f}%')