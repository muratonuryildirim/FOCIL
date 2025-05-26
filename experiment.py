import time
import numpy as np
from dm import DataManager
from cil_learner import Learner
from utils import print_and_log, setup

def multiple_run(args):

    apt=np.array([])
    acc=np.array([])
    bwt=np.array([])
    run_time=np.array([])
    for run in range(args.num_runs):
        setup(args) 
        data_manager = DataManager(args.dataset, 
                                    shuffle=True, 
                                    seed=args.seed, 
                                    init_cls=args.init_classes, 
                                    increment=args.num_classes_per_task)
        learner = Learner(args, data_manager)

        t0=time.time()
        for task in range(args.num_tasks):
            learner.train(task)
        t1=time.time()
        #print_and_log('\nTotal Training Time= {:.2f} min'.format((t1-t0)/60))

        #learner.evaluate()
        learner.ensemble_evaluation_majority()
        apt_run, acc_run, bwt_run = learner.report_scores()
        
        apt = np.append(apt, apt_run)
        acc = np.append(acc, acc_run)
        bwt = np.append(bwt, bwt_run)
        run_time = np.append(run_time, (t1-t0)/60)

        print_and_log('\n Run {} is done.'.format(run))
        
    apt=apt.reshape(-1,args.num_tasks)
    acc=acc.reshape(-1,args.num_tasks)

    average_apt = np.array2string(np.mean(apt, axis=0),separator=', ') 
    std_apt = np.array2string(np.std(apt, axis=0),separator=', ')

    average_acc = np.mean(acc, axis=0)
    std_acc = np.std(acc, axis=0)

    final_acc_mean=average_acc[-1]
    final_acc_std=std_acc[-1]

    average_acc = np.array2string(average_acc, separator=', ')
    std_acc = np.array2string(std_acc, separator=', ')

    average_bwt = np.mean(bwt)
    std_bwt = np.std(bwt)

    average_t = np.mean(run_time)
    std_t = np.std(run_time)

    print_and_log('~' * 60)
    print_and_log('sparsity: {}, lr: {}, window: {}, threshold: {}'.format(args.sparsity, args.lr, args.window_size, args.drift_threshold))
    print_and_log('Experiments are over with {} runs.'.format(args.num_runs))
    print_and_log('\nAPT: {}'.format(average_apt))
    print_and_log('ACC: {}'.format(average_acc))
    print_and_log('\nFinal ACC: {:.2f} +- {:.2f}'.format(final_acc_mean, final_acc_std))
    print_and_log('Final BWT: {:.2f} +- {:.2f}'.format(average_bwt, std_bwt))
    print_and_log('\nTime taken (min): {:.2f} +- {:.2f}'.format(average_t, std_t))
    print_and_log('~' * 60)

        
