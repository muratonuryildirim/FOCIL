<div align="center">
<img src="./resources/focil-icon.png" width="225px">
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2403.14684">Paper</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#cite">Cite</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="https://muratonuryildirim.github.io">Contact</a>
</p>

# FOCIL: Finetune-and-Freeze for Online Class Incremental Learning by Training Randomly Pruned Sparse Experts
*FOCIL is a method proposed for Online Class-Incremental Learning. It fine-tunes the main backbone continually by training a randomly pruned sparse subnetwork for each task. Then, it freezes the trained connections to prevent forgetting. FOCIL also determines the sparsity level and learning rate per task adaptively, and ensures nearly zero forget across all tasks without expanding the network or storing replay data.* 

<div align="center">
<img src="./resources/focil-results.webp">
</div>

## How to Use

For example, to train 20-Task CIFAR100, run:

```
python main.py
       --dataset cifar100
       --num_classes 100
       --num_tasks 20
       --num_classes_per_task 5
       --backbone resnet18
```


## License

Please check the [MIT license](./LICENSE) that is listed in this repository.

## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [In-Time-Over-Parameterization](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization)
- [Optuna](https://github.com/optuna/optuna)
- [PyCIL](https://github.com/G-U-N/PyCIL)
