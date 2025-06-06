import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import resnet20  # from kuangliu/pytorch-cifar
from torchvision.models import vgg11

# PGD Attack implementation
def pgd_attack(model, images, labels, eps, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    ori_images = images.data

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

# White-box test renamed and expanded
def white_box_test(model, device, test_loader, epsilons, alpha, iters):
    accuracies = []
    examples = []

    for eps in epsilons:
        correct = 0
        adv_samples = []

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            perturbed_data = pgd_attack(model, data, target, eps, alpha, iters)
            output = model(perturbed_data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            if len(adv_samples) < 5:
                adv_samples.append((data[0].cpu(), perturbed_data[0].cpu(), pred[0].item(), target[0].item()))

        acc = correct / len(test_loader.dataset)
        accuracies.append(acc)
        examples.append(adv_samples)
        print(f"[White-box] Epsilon={eps:.3f}  Accuracy={acc:.4f}")

    # Plot accuracy vs epsilon
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, marker='o')
    plt.title("White-box PGD Attack - Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("whitebox_accuracy_plot.png")
    plt.close()

    # Save example images
    for i, eps in enumerate(epsilons):
        fig, axs = plt.subplots(1, len(examples[i]), figsize=(12,3))
        for j, (orig, adv, pred, label) in enumerate(examples[i]):
            axs[j].imshow(adv.permute(1, 2, 0))
            axs[j].set_title(f"True:{label}, Pred:{pred}")
            axs[j].axis('off')
        plt.suptitle(f"Epsilon: {eps:.3f}")
        plt.savefig(f"adv_examples_eps_{int(eps*255)}.png")
        plt.close()

    return accuracies

def black_box_test(surrogate_model, target_model, device, test_loader, epsilon, alpha, iters):
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = pgd_attack(surrogate_model, data, target, epsilon, alpha, iters)
        output = target_model(adv_data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    acc = correct / total
    print(f"[Black-box] PGD Transfer Attack: Epsilon={epsilon}, Accuracy={correct}/{total} = {acc:.4f}")
    return acc

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load ResNet-20 target model
resnet20_ckpt = r"D:\neural_network\作业4\Resnet20\resnet20-12fca82f.th"
resnet20_model = resnet20().to(device)
resnet20_model.load_state_dict(torch.load(resnet20_ckpt, map_location=device))
resnet20_model.eval()

# Load surrogate model (VGG11)
vgg11_ckpt = r"D:\neural_network\作业4\VGG11\vgg11_cifar10.pth"
surrogate_model = vgg11(num_classes=10).to(device)
surrogate_model.load_state_dict(torch.load(vgg11_ckpt, map_location=device))
surrogate_model.eval()

# Run white-box attack on surrogate model
epsilons = [0, 2/255, 4/255, 6/255, 8/255]
alpha = 2/255
iters = 10

white_box_test(surrogate_model, device, test_loader, epsilons, alpha, iters)
black_box_test(surrogate_model, resnet20_model, device, test_loader, epsilon=8/255, alpha=2/255, iters=10)
