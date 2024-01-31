from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset
import os
import shutil
import requests
import zipfile
import io
import random

''' DATASETS '''
class iMNIST:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        else:
            self.mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.mnist):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.mnist[self.split_datasets[self.task_id][idx]]
    

class iFashionMNIST:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.fashionmnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        else:
            self.fashionmnist = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.fashionmnist):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.fashionmnist[self.split_datasets[self.task_id][idx]]
    
    
class iEMNIST_letters:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.emnist = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
        else:
            self.emnist = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.emnist):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.emnist[self.split_datasets[self.task_id][idx]]


class iCIFAR10:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        else:
            self.cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.cifar10):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.cifar10[self.split_datasets[self.task_id][idx]]


class iCIFAR100:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        else:
            self.cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.cifar100):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.cifar100[self.split_datasets[self.task_id][idx]]


class iTinyImageNet(Dataset):
    def __init__(self, root_dir, train=True, transform=None, tasks=None):

        if not os.path.exists('tiny-imagenet-200'):
            print('Downloading the dataset...')
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()

        self.root_dir = root_dir
        self.DIR = os.path.join(root_dir, 'train') if train else os.path.join(root_dir, 'val')

        if not train:
            if os.path.isfile(os.path.join(self.DIR, 'val_annotations.txt')):
                fp = open(os.path.join(self.DIR, 'val_annotations.txt'), 'r')
                data = fp.readlines()
                val_img_dict = {}  # dict {.jpg:[class_name]}
                for line in data:
                    words = line.split('\t')
                    val_img_dict[words[0]] = words[1]
                fp.close()
                for img, folder in val_img_dict.items():
                    newpath = (os.path.join(self.DIR, folder, 'images'))
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    if os.path.exists(os.path.join(self.DIR, 'images', img)):
                        os.rename(os.path.join(self.DIR, 'images', img), os.path.join(newpath, img))
            if os.path.exists(os.path.join(self.DIR, 'images')):
                os.rmdir(os.path.join(self.DIR, 'images'))
            if os.path.exists(os.path.join(self.DIR, 'val_annotations.txt')):
                os.remove(os.path.join(self.DIR, 'val_annotations.txt'))

        self.transform = transform
        self.tasks = tasks
        self.task_id = 0
        self.classes = os.listdir(self.DIR)  # list [class_name]
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}  # dict {class_name:[class_id]}
        self.class_files = {class_id: os.listdir(os.path.join(self.DIR, class_name, 'images'))
                            for class_name, class_id in self.class_to_id.items()}  # dict {class_id:[.jpg]}
        self.task_imgs = {}  # dict {task_id:[.jpg]}
        self.task_class_ids = {}  # dict {task_id:[class_id]}
        for task_no, class_ids in enumerate(tasks):
            samples_list = []
            class_list = []
            for class_id in class_ids:
                samples_list.extend(self.class_files[class_id])
                class_list.extend([class_id] * len(self.class_files[class_id]))
            self.task_imgs[task_no] = samples_list
            self.task_class_ids[task_no] = class_list

    def __len__(self):
        return len(self.task_class_ids[self.task_id])

    def __getitem__(self, idx):
        img_name = self.task_imgs[self.task_id][idx]
        folder_name = next((self.classes[class_id] for class_id, imgs in self.class_files.items() if img_name in imgs), None)
        img_path = os.path.join(self.DIR, folder_name, 'images', img_name)
        image = Image.open(img_path)
        class_id = self.task_class_ids[self.task_id][idx]
        if self.transform:
            image= image.convert('RGB')
            image = self.transform(image)
        return image, class_id

    def set_task(self, task_id):
        self.task_id = task_id

class iMiniImageNet(Dataset):
    def __init__(self, root_dir, train=True, transform=None, tasks=None):
        self.root_dir = root_dir
        self.DIR = os.path.join(root_dir, 'train') if train else os.path.join(root_dir, 'validation')

        if not os.path.exists(self.root_dir):
            with zipfile.ZipFile('data/miniImageNet100.zip', "r") as zip_ref:
                # Replace "path/to/dataset" with the path where you want to extract the files
                zip_ref.extractall(self.root_dir)

        if not os.path.exists(os.path.join(self.root_dir, 'validation')):
            validation_dir = os.path.join(self.root_dir, 'validation')
            # validation_fraction = 0.20
            # Loop through each folder in the data directory
            for folder_name in os.listdir(self.root_dir):
                folder_path = os.path.join(self.root_dir, folder_name)
                if os.path.isdir(folder_path):
                    # Create a validation folder for this folder
                    validation_folder = os.path.join(validation_dir, folder_name)
                    os.makedirs(validation_folder, exist_ok=True)

                    # Get a list of all the files in the folder
                    file_names = os.listdir(folder_path)
                    num_validation_files = 100
                    # num_validation_files = int(len(file_names) * validation_fraction)
                    random.shuffle(file_names)
                    # Move the first num_validation_files to the validation folder
                    for file_name in file_names[:num_validation_files]:
                        src_path = os.path.join(folder_path, file_name)
                        dst_path = os.path.join(validation_folder, file_name)
                        shutil.move(src_path, dst_path)

        if not os.path.exists(os.path.join(self.root_dir, 'train')):
            train_dir = os.path.join(self.root_dir, 'train')
            # Create the train directory if it doesn't exist
            os.makedirs(train_dir, exist_ok=True)

            # Loop through each directory in the data directory
            for dir_name in os.listdir(self.root_dir):
                dir_path = os.path.join(self.root_dir, dir_name)
                if os.path.isdir(dir_path) and dir_name != "validation" and dir_name != "train":
                    # Move the directory to the train directory
                    new_dir_path = os.path.join(train_dir, dir_name)
                    shutil.move(dir_path, new_dir_path)

        self.transform = transform
        self.tasks = tasks
        self.task_id = 0
        self.classes = os.listdir(self.DIR)  # list [class_name]
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}  # dict {class_name:[class_id]}
        self.class_files = {class_id: os.listdir(os.path.join(self.DIR, class_name))
                            for class_name, class_id in self.class_to_id.items()}  # dict {class_id:[.jpg]}
        self.task_imgs = {}  # dict {task_id:[.jpg]}
        self.task_class_ids = {}  # dict {task_id:[class_id]}
        for task_no, class_ids in enumerate(tasks):
            samples_list = []
            class_list = []
            for class_id in class_ids:
                samples_list.extend(self.class_files[class_id])
                class_list.extend([class_id] * len(self.class_files[class_id]))
            self.task_imgs[task_no] = samples_list
            self.task_class_ids[task_no] = class_list

    def __len__(self):
        return len(self.task_class_ids[self.task_id])

    def __getitem__(self, idx):
        img_name = self.task_imgs[self.task_id][idx]
        folder_name = next((self.classes[class_id] for class_id, imgs in self.class_files.items() if img_name in imgs), None)
        img_path = os.path.join(self.DIR, folder_name, img_name)
        image = Image.open(img_path)
        class_id = self.task_class_ids[self.task_id][idx]
        if self.transform:
            image = image.convert('RGB')
            image = self.transform(image)
        return image, class_id

    def set_task(self, task_id):
        self.task_id = task_id

