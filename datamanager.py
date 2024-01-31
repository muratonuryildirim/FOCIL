from datasets import iCIFAR10, iMNIST, iFashionMNIST, iEMNIST_letters, iMiniImageNet, iCIFAR100, iTinyImageNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

class DataManager():
    def __init__(self, args):
        self.dataset = args.dataset
        self.batch_train = args.batch_train
        self.batch_test = args.batch_test
        self.batch_mask = args.batch_mask
        self.num_tasks = args.num_tasks
        self.num_classes = args.num_classes
        self.num_classes_per_task = args.num_classes_per_task
        self.train_dataset = self._task_constructor(self.dataset, train=True)
        self.test_dataset = self._task_constructor(self.dataset, train=False)
        
    def _task_constructor(self, dataset, train=True):

        tasks_order = np.arange(self.num_classes).reshape((self.num_tasks, self.num_classes_per_task))

        if dataset == 'mnist':
            mean, std = (0.1307,), (0.3081,)
            transform = self._get_transform(mean=mean, std=std)
            dataset = iMNIST(train=train, transform=transform, tasks=tasks_order)
        if dataset == 'fashionmnist':
            mean, std = (0.1307,), (0.3081,)
            transform = self._get_transform(mean=mean, std=std)
            dataset = iFashionMNIST(train=train, transform=transform, tasks=tasks_order)
        if dataset == 'emnist':
            mean, std = (0.1736,), (0.3248,)
            transform = self._get_transform(size=28, padding=4, mean=mean, std=std, preprocess=True)
            dataset = iEMNIST_letters(train=train, transform=transform, tasks=tasks_order)
        if dataset == 'cifar10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            transform = self._get_transform(size=32, padding=4, mean=mean, std=std, preprocess=True)
            dataset = iCIFAR10(train=train, transform=transform, tasks=tasks_order)
        if dataset == 'cifar100':
            mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            transform = self._get_transform(size=32, padding=4, mean=mean, std=std, preprocess=True)
            dataset = iCIFAR100(train=train, transform=transform, tasks=tasks_order)
        if dataset == 'tinyImagenet200':
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            transform = self._get_transform(size=64, padding=4, mean=mean, std=std, preprocess=True)
            dataset = iTinyImageNet('tiny-imagenet-200', train=train, transform=transform, tasks=tasks_order)
        if dataset == 'miniImagenet100':
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            transform = self._get_transform(size=32, padding=4, mean=mean, std=std, preprocess=True)
            dataset = iMiniImageNet('data/miniImageNet100', train=train, transform=transform, tasks=tasks_order)

        return dataset


    def _get_transform(self, size=None, padding=None, mean=None, std=None, preprocess=False):
        transform = []
        if preprocess:
            transform.append(transforms.Resize(size))
            transform.append(transforms.RandomCrop(size, padding=padding))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))

        return transforms.Compose(transform)
    
    def get_loader(self, dataset, task, mode='train'):
        dataset.set_task(task)
        if mode == 'train':
            loader = DataLoader(dataset, batch_size=self.batch_train, shuffle=True)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_test, shuffle=True)
        return loader
    
    def get_mask_selection_loader(self, dataset, task):
        dataset.set_task(task)
        loader = DataLoader(dataset, batch_size=self.batch_mask, shuffle=True)
        return loader
