import argparse
import time
import optuna
from optuna.samplers import RandomSampler
from optuna.pruners import SuccessiveHalvingPruner
from datamanager import DataManager
from cil_learner import Learner
from utils import setup, print_and_log

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset to use')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='number of classes')
    parser.add_argument('--num_tasks', type=int, default=20,
                        help='number of tasks')
    parser.add_argument('--num_classes_per_task', type=int, default=5,
                        help='number of classes per task')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='network architecture to use')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number training epochs')
    parser.add_argument('--batch_train', type=int, default=10,
                        help='number of examples per training batch')
    parser.add_argument('--batch_test', type=int, default=256,
                        help='number of examples per test batch')
    parser.add_argument('--batch_mask', type=int, default=10,
                        help='number of examples to identify the expert')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='sgd momentum')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='weight decay coefficient')
    parser.add_argument('--regularize', type=bool, default=False,
                        help='regularize weights of the previous tasks')
    parser.add_argument('--isolate', type=bool, default=True,
                        help='freeze weights of the previous tasks')
    parser.add_argument('--fixed_topology', type=bool, default=True,
                        help='controls update of the topology')
    parser.add_argument('--mask_selection', type=str, default='max',
                        help='mask selection method. options: max')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer. options: adam, sgd')
    
    args = parser.parse_args()
    setup(args)
    data_manager = DataManager(args)
    learner = Learner(args, data_manager)
    
    t0=time.time()
    for task in range(args.num_tasks):
        study = optuna.create_study(sampler=RandomSampler(),
                                    pruner=SuccessiveHalvingPruner(),
                                    direction="maximize",
                                    study_name='{}_task{}_{}'.format(args.dataset, task, args.backbone),
                                    #storage='sqlite:///{}_{}task_{}.db'.format(args.dataset, args.num_tasks, args.backbone)
                                    )
        study.optimize(learner.train, n_trials=20)
        learner.after_task()  
    t1=time.time()
    print_and_log('\nTotal Training Time= {:.2f} min'.format((t1-t0)/60))
    
    learner.evaluate()
    learner.report_scores()

if __name__ == '__main__':
    main()
