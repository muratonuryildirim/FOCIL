# FOCIL: Finetune-and-Freeze for Online Class Incremental Learning by Training Randomly Pruned Sparse Experts
*Class incremental learning (CIL) in an online continual learning setting strives to acquire knowledge on a series of novel classes from a data stream, using each data point only once for training which is more realistic compared to offline mode where it is assumed all data from novel class(es) is readily available. Current online CIL approaches store a subset of the previous data which creates heavy overhead costs in terms of both memory and computation, as well as privacy issues. In this paper, we propose a new online CIL approach called FOCIL. It fine-tunes the main backbone continually by training a randomly pruned sparse subnetwork for each task. Then, it freezes the trained connections to prevent forgetting. FOCIL also determines the sparsity level and learning rate per task adaptively, and ensures (almost) zero forget across all tasks without storing any replay data. Experimental results on 10-Task CIFAR100, 20-Task CIFAR100 and 100-Task TinyImagenet, demonstrate that our method outperforms SOTAs by a large margin.* 


## Training

Here, we provide parsing examples to train FOCIL.

To train 10-Task CIFAR100 *(total number of classes: 100, number of classes per task: 10)*
```
python main.py
       --dataset cifar100
       --num_classes 100
       --num_tasks 10
       --num_classes_per_task 10
       --backbone resnet18
```

To train 20-Task CIFAR100 *(total number of classes: 100, number of classes per task: 5)*
```
python main.py
       --dataset cifar100
       --num_classes 100
       --num_tasks 20
       --num_classes_per_task 5
       --backbone resnet18
```

To train 100-Task TinyImageNet *(total number of classes: 200, number of classes per task: 2)*
```
python main.py
       --dataset tinyImagenet200
       --num_classes 200
       --num_tasks 100
       --num_classes_per_task 10
       --backbone resnet18
```

To train 5-Task MNIST *(total number of classes: 10, number of classes per task: 2)*
```
python main.py
       --dataset mnist
       --num_classes 10
       --num_tasks 5
       --num_classes_per_task 2
       --backbone mlp
```

To train 5-Task Fashion-MNIST *(total number of classes: 10, number of classes per task: 2)*
```
python main.py
       --dataset fashionmnist
       --num_classes 10
       --num_tasks 5
       --num_classes_per_task 2
       --backbone mlp
```
