import copy
import time
import torch
import optuna
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from masking import Masking
from backbones import ResNet18, VGG16, MobileNetV2, MLP_300_100
from utils import print_and_log

class Learner():
    def __init__(self, args, data_manager):
        self.used_params = {}
        self.cur_task = 0
        self._known_classes = 0
        self.data_manager = data_manager
        self.l2 = args.l2
        self.momentum = args.momentum
        self.epochs = args.epochs
        self.optimizer = args.optimizer
        self.backbone = args.backbone
        self.isolate = args.isolate
        self.regularize = args.regularize
        self.fixed_topology = args.fixed_topology
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NET = self._get_backbone(self.backbone, self.data_manager.num_classes)
        
        self.total_params = sum(p.numel() for p in self.NET.parameters())
        self.best_acc = -1.0

        self.acc_matrix = np.zeros((self.data_manager.num_tasks, self.data_manager.num_tasks))
        self.bwt = np.zeros(self.data_manager.num_tasks)
        self.fwt = np.zeros(self.data_manager.num_tasks)
    
    def after_task(self):
        self.net.load_state_dict(torch.load(f'./models/{self.data_manager.dataset}.pt'))
        self._update_used_params()
        self._stack_subnets()
        self._calculate_nonzeros()
        self.cur_task += 1
        self.best_acc = -1.0
        self._known_classes += self.data_manager.num_classes_per_task

    def train(self, trial):
        print_and_log('\nTraining...')
        
        self.net = self._get_backbone(self.backbone, self.data_manager.num_classes)
        if self.cur_task !=0:
            self.net.load_state_dict(torch.load(f'./models/{self.data_manager.dataset}_DENSE.pt'))
        
        self.train_loader = self.data_manager.get_loader(self.data_manager.train_dataset, self.cur_task)
        #self.test_loader = self.data_manager.get_loader(self.data_manager.test_dataset, self.cur_task)

        lr = trial.suggest_float('lr', 0.05, 0.3, step = 0.05)
        density = trial.suggest_float('density', 0.02, 0.1, step = 0.02)
 
        optimizer, lr_scheduler = self._set_optimizer_and_scheduler(lr)
        self.mask = Masking(net=self.net, density=density, sparse_init="erk", device=self.device)

        for epoch in range(self.epochs):
            task_t0 = time.time()
            train_loss = 0
            correct = 0
            n = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                batch_t0 = time.time()
                self.net.train()
                inputs, targets = inputs.to(self.device), targets.to(torch.int64).to(self.device)
                logits = self.net(inputs)
                dummy_targets = targets - self._known_classes
                loss = F.cross_entropy(logits[:, self._known_classes : (self._known_classes + self.data_manager.num_classes_per_task)], dummy_targets)

                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                
                self._freeze_used_params()
                optimizer.step()
                self.mask.step(epoch)
                
                preds = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += preds.eq(targets.view_as(preds)).sum().item()
                n += len(targets)
                
                if batch_idx % 50 == 0:
                    batch_t1 = time.time()
                    batch_acc = 100 * correct / n
                    #test_acc = self._compute_accuracy(self.test_loader)

                    print_and_log('Batch {:3d} | Train: acc={:5.2f}% | time={:5.1f}ms |'.format(batch_idx, batch_acc, 1000*(batch_t1-batch_t0)))
                    #print(' Test: acc={:5.2f}% |'.format(test_acc), end='')

                    trial.report(batch_acc, batch_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            task_t1 = time.time()
            lr_scheduler.step()

            train_acc = 100 * correct / n
            #test_acc = self._compute_accuracy(self.test_loader)
            print_and_log('\nSummary => Train: loss={:.3f}, acc={:5.2f}% | time={:5.1f}s |'.format(train_loss, train_acc, (task_t1-task_t0)))
            #print(' Test: acc={:5.2f}% |\n'.format(test_acc))  

            if train_acc > self.best_acc:
                self.best_acc = train_acc
                self._save_checkpoint()
                self._save_mask()
            
        return train_acc
    
    def evaluate(self):
        print_and_log('\nTesting...')
        self.NET.load_state_dict(torch.load('./models/{0}_DENSE.pt'.format(self.data_manager.dataset)))
        
        #acc_matrix
        for task in range(self.cur_task):
            for t in range(task+1):
                self.test_loader = self.data_manager.get_loader(self.data_manager.test_dataset, t)
                t0=time.time()
                selected_mask = self._mask_selection(t)
                t1=time.time()
                self.net = self._set_mask(selected_mask)
                test_accuracy = self._compute_accuracy(self.test_loader)
                self.acc_matrix[task, t] = test_accuracy
                mask_time=(t1-t0)
            print_and_log('Time Spent to Identify the Expert(s) at task {}: {:.2f} sec '.format(task, mask_time))

    def report_scores(self):
        print_and_log('\nACCURACY MATRIX:\n {}'.format(self.acc_matrix))   
        #bwt_matrix
        for task in range(self.cur_task-1):
            self.bwt[task] = self.acc_matrix[self.cur_task-1, task] - self.acc_matrix[task, task]
        print_and_log('\nBWT:\n {}'.format(self.bwt))

        #inc_matrix
        final_acc = self.acc_matrix[-1]
        inc_acc = np.zeros_like(final_acc)
        for i in range(len(final_acc)):
            inc_acc[i] = final_acc[i] if i == 0 else np.mean(final_acc[:i+1])
        print_and_log('\nINCREMENTAL ACCURACY:\n {}'.format(inc_acc))
        print_and_log('\nAverage BWT:\n {}'.format(np.mean(self.bwt[:-1])))

    def _compute_accuracy(self, loader, for_mask=False):
        self.net.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(torch.int64).to(self.device)
                logits = self.net(inputs)
                preds = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += preds.eq(targets.view_as(preds)).sum().item()
                n += len(targets)
                if for_mask:
                    break

        return np.around(100 * correct / n, decimals=2)

    def _set_optimizer_and_scheduler(self, lr):
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.l2, nesterov=True)
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=self.l2)
        else:
            print_and_log('Unknown optimizer: {0}'.format(optimizer))
            raise Exception('Unknown optimizer.')

        learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[int(self.epochs * 1/2), int(self.epochs * 3/4)])

        return optimizer, learning_rate_scheduler

    def _get_backbone(self, backbone, num_classes):
        if backbone == 'resnet18':
           net = ResNet18(num_classes).to(self.device)
        elif backbone == 'mobilenetv2':
            net = MobileNetV2(num_classes).to(self.device)
        elif backbone == 'vgg16':
            net = VGG16('like', num_classes).to(self.device)
        elif backbone == 'mlp':
            net = MLP_300_100(num_classes).to(self.device)
        return net
    
    def _update_used_params(self):
        current_mask = torch.load(f'./masks/{self.data_manager.dataset}_task{self.cur_task}_mask.pt')
        self.used_params = {k: self.used_params.get(k, 0) + current_mask.get(k, 0) for k in set(self.used_params) | set(current_mask)}

    def _calculate_nonzeros(self):
        non_zero_count = 0
        for nonzeros in self.used_params.values():
            non_zero_count += torch.count_nonzero(nonzeros)
        print_and_log('\nOverall Sparsity: {:.4f}.'.format(non_zero_count/self.total_params))
        print_and_log('=' * 60)

    def _freeze_used_params(self):
        if self.cur_task != 0 and self.isolate:
            for name, params in self.net.named_parameters():
                if name in self.mask.masks:
                    params.grad[self.used_params[name] != 0] = 0

    def _stack_subnets(self):
        if self.cur_task == 0:
            self.NET = copy.deepcopy(self.net)
        else:
            for (name, param), (old_name, old_param) in zip(self.net.named_parameters(), self.NET.named_parameters()):
                param.data[param == 0] = old_param.data[param == 0]
                self.NET = copy.deepcopy(self.net)
        torch.save(self.NET.state_dict(), './models/{0}_DENSE.pt'.format(self.data_manager.dataset))

    def _set_mask(self, task_id):
        task_mask = torch.load(f'./masks/{self.data_manager.dataset}_task{task_id}_mask.pt')
        masked_net = copy.deepcopy(self.NET)
        for n, t in masked_net.named_parameters():
            if n in task_mask: t.data = t.data * task_mask[n]
        return masked_net

    def _save_checkpoint(self):
        torch.save(self.net.state_dict(), './models/{0}.pt'.format(self.data_manager.dataset))

    def _save_mask(self):
        torch.save(self.mask.masks, './masks/{0}_task{1}_mask.pt'.format(self.data_manager.dataset, self.cur_task))

    def _mask_selection(self, task):
        mask_selection_loader = self.data_manager.get_mask_selection_loader(self.data_manager.test_dataset, task)
        #t0=time.time()
        max_out = []
        for mask in range(task+1):
            self.net = self._set_mask(mask)
            mask_acc = self._compute_accuracy(mask_selection_loader, for_mask=True)
            max_out.append(mask_acc)
        selected_mask = np.argmax(max_out)
        #t1=time.time()
        #print_and_log('Time Spend to Identify the Expert(s) at task {}: {} sec '.format(task, (t1-t0)))
        return selected_mask
