import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import ResNet18
from autoattack import AutoAttack

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint .pth')
parser.add_argument('--key', type=str, default='net', help='state dict key in checkpoint (net or ema)')
parser.add_argument('--n_examples', type=int, default=10000, help='number of test examples to attack')
parser.add_argument('--version', type=str, default='standard', choices=['standard', 'rand', 'fast'],
                    help='fast = apgd-ce + apgd-t only (good proxy, much cheaper)')
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


class NormalizedModel(nn.Module):
    """AutoAttack works in [0,1] image space; normalization must live inside forward."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


# IMPORTANT: no Normalize here — AutoAttack needs raw [0,1] inputs
transform_test = transforms.Compose([transforms.ToTensor()])

testset = DATASET_CLS(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

net = ResNet18(num_classes=NUM_CLASSES).to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load(args.ckpt, map_location=device)
net.load_state_dict(checkpoint[args.key])
net.eval()

model = NormalizedModel(net, CIFAR_MEAN, CIFAR_STD).to(device)
model.eval()

x_all, y_all = [], []
for inputs, targets in testloader:
    x_all.append(inputs)
    y_all.append(targets)

x_test = torch.cat(x_all, dim=0)[:args.n_examples].to(device)
y_test = torch.cat(y_all, dim=0)[:args.n_examples].to(device)

if args.version == 'fast':
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom',
                           attacks_to_run=['apgd-ce', 'apgd-t'])
else:
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version=args.version)

x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)

aa_correct = 0
with torch.no_grad():
    for i in range(0, x_adv.size(0), 100):  # batched: full-set forward OOMs on 8GB
        outputs_adv = model(x_adv[i:i + 100])
        aa_correct += outputs_adv.argmax(1).eq(y_test[i:i + 100]).sum().item()

aa_acc = 100.0 * aa_correct / y_test.size(0)
print(f'AutoAttack ({args.version}, n={args.n_examples}) Acc: {aa_acc:.2f}%')
