import argparse
from dm import DataManager
from learner import Learner
from utils import setup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_tasks', type=int, default=5,  help='number of tasks')
    parser.add_argument('--init_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--num_classes_per_task', type=int, default=2, help='number of classes per task')
    parser.add_argument('--epochs', type=int, default=1, help='number training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
    parser.add_argument('--l2', type=float, default=5e-4, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=10, help='number of data per batch')
    parser.add_argument('--backbone', type=str, default='resnet18', help='network architecture to use')
    parser.add_argument('--sparsity', type=float, default=0.95, help='amount of connections to remove from the network')
    parser.add_argument('--window_size', type=int, default=10, help='method to use for sample sparsification')
    parser.add_argument('--drift_threshold', type=float, default=0.5, help='method to use for sample sparsification')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    setup(args) 
    data_manager = DataManager(args.dataset, 
                               shuffle=True,
                               seed=args.seed,
                               init_cls=args.init_classes,
                               increment=args.num_classes_per_task)
    learner = Learner(args, data_manager)
    for task in range(args.num_tasks):
        learner.train(task)
    learner.ensemble_evaluation()

if __name__ == '__main__':
    main()
