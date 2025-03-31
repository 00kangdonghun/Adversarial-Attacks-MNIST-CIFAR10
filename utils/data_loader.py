import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64):
    # 이미지 데이터를 텐서로 변환 (0~1 범위로 정규화됨)
    transform = transforms.ToTensor()
    # 학습 데이터셋 로드 (자동 다운로드)
    train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    # 테스트 데이터셋 로드
    test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    # DataLoader로 묶어서 반환
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)

def get_cifar10_loaders(batch_size=64):
    # CIFAR-10 이미지를 텐서로 변환
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 학습 데이터셋 로드
    train = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    # 테스트 데이터셋 로드
    test = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    # DataLoader로 묶어서 반환
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)
