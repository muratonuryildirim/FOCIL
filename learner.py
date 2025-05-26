import copy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from model_sparsity import ModelSparsifier
from backbones import get_backbone
from utils import print_and_log
from collections import Counter

class Learner():
    def __init__(self, args, data_manager):
        self.model_sparsity = args.sparsity
        self.epochs = args.epochs
        self.lr = args.lr
        self.l2 = args.l2
        self.momentum = args.momentum
        self.data_manager = data_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_tasks = args.num_tasks
        self.num_classes = args.num_classes
        self.num_classes_per_task = args.num_classes_per_task

        self.NET = get_backbone(args.backbone, self.num_classes, self.batch_size, self.device)
        self.net = get_backbone(args.backbone, self.num_classes, self.batch_size, self.device)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.l2, nesterov=True)
        self.model_sparsifier = ModelSparsifier(net=self.net, sparsity=self.model_sparsity, sparse_init="erk", device=self.device)

        # drift dectector
        self.acc_window_length=args.window_size
        self.threshold = args.drift_threshold
        self.acc_window=[]

        #stats
        self.cur_task = 0
        self.used_params = {}
        self.total_params = sum(p.numel() for p in self.NET.parameters())

        # results
        self.acc_matrix = np.zeros((self.num_tasks, self.num_tasks))
        self.bwt = np.zeros(self.num_tasks)
        self.fwt = np.zeros(self.num_tasks)


    def train(self, task):
        train_dataset = self.data_manager.get_dataset(np.arange(task*self.num_classes_per_task, (task+1)*self.num_classes_per_task), source= 'train', mode= 'train')
        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        print_and_log('\nTraining started...' if task == 0 else '\nNew task introduced...')

        task_t0 = time.time()
        for epoch in range(self.epochs):
            train_loss = 0
            correct = 0
            n = 0
            for batch_idx, batch in enumerate(self.train_loader):
                batch_t0 = time.time()
                _, inputs, targets = batch
                self.net.train()
                self.inputs, self.targets = inputs.to(self.device), targets.to(torch.int64).to(self.device)
                self.logits = self.net(self.inputs)
                self.loss =  F.cross_entropy(self.logits, self.targets)
                preds = self.logits.argmax(dim=1, keepdim=True)
                correct += preds.eq(self.targets.view_as(preds)).sum().item()
                n += len(self.targets)
                self.batch_acc = 100 * correct / n

                self._drift_detector()

                self.optimizer.zero_grad()
                self.loss.backward()
                train_loss += self.loss.item()
                
                self._freeze_used_params()
                self.optimizer.step()
                self.model_sparsifier.apply()

                if batch_idx % 50 == 0:
                    batch_t1 = time.time()
                    print_and_log('Batch {:3d} | Train: loss={:5.2f} | acc={:5.2f}% | time={:6.1f}ms |'.format(batch_idx, self.loss, self.batch_acc, 1000*(batch_t1-batch_t0)))

        task_t1 = time.time()
        #lr_scheduler.step()

        train_acc = 100 * correct / n
        print_and_log('\nSummary => Train: loss={:.3f} | acc={:5.2f}% | time={:6.1f}s |'.format(train_loss/batch_idx, train_acc, (task_t1-task_t0)))

        if task == self.num_tasks-1:
            print_and_log('Data Stream is Over.')
            self._after_task()

    def _drift_detector(self):
        self.acc_window = np.append(self.acc_window, self.batch_acc)
        if len(self.acc_window)>self.acc_window_length: 
            self.acc_window=np.delete(self.acc_window, 0)
            self.acc_window_mean=np.mean(self.acc_window)
            if (self.batch_acc < self.acc_window_mean * self.threshold):
                print_and_log('New peak found...')
                self._after_task()
                self.logits = self.net(self.inputs)
                self.loss = F.cross_entropy(self.logits, self.targets)

    def _after_task(self):
        self._save_checkpoint()
        self._save_mask()
        self.net.load_state_dict(torch.load(f'./models/{self.dataset}.pt'))
        self._update_used_params()
        self._stack_subnets()
        self._calculate_nonzeros()

        self.cur_task += 1
        self.acc_window=np.array([])
        self.running_loss_window_mean=np.array([0])
        self.running_loss_window_variance=np.array([0])

        self.net.load_state_dict(torch.load(f'./models/{self.dataset}_DENSE.pt'))
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.l2, nesterov=True)
        self.model_sparsifier = ModelSparsifier(net=self.net, sparsity=self.model_sparsity, sparse_init="erk", device=self.device)

    def ensemble_evaluation(self):
        print_and_log('\nEnsemble Evaluation...')
        self.NET.load_state_dict(torch.load('./models/{0}_DENSE.pt'.format(self.dataset)))
        
        ensemble_correct, ensemble_n = 0, 0
        all_accuracy = 0
        with torch.no_grad():
            for task in range(self.num_tasks):
                self.test_dataset = self.data_manager.get_dataset(np.arange(task*self.num_classes_per_task, (task+1)*self.num_classes_per_task), source='test', mode='test')
                self.test_loader = DataLoader(self.test_dataset, 512, shuffle=True)
                
                for batch_idx, (_, inputs, targets) in enumerate(self.test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(torch.int64).to(self.device)
                    all_preds = []
                    weights = []
                    
                    for mask in range(task + 1):
                        self.net = self._set_mask(mask)
                        self.net.eval()
                        logits = self.net(inputs)
                        logits = F.softmax(logits, dim=1)
                        preds = logits.argmax(dim=1, keepdim=True)
                        all_preds.append(preds)
                        weights.append(mask + 1)
                    
                    # Majority voting
                    all_preds = torch.cat(all_preds, dim=1)
                    majority_preds = []
                    for i in range(all_preds.size(0)):
                        votes = all_preds[i].tolist()
                        weighted_votes = Counter()
                        for j, vote in enumerate(votes):
                            weighted_votes[vote] += weights[j]
                        majority_vote = weighted_votes.most_common(1)[0][0]
                        majority_preds.append(majority_vote)
                    
                    majority_preds = torch.tensor(majority_preds).to(self.device)
                    ensemble_correct += majority_preds.eq(targets).sum().item()
                    ensemble_n += len(targets)

                ensemble_accuracy = np.around(100 * ensemble_correct / ensemble_n, decimals=2)
                all_accuracy += ensemble_accuracy
                print_and_log('Stream {} Ensemble Accuracy: {:.2f}%'.format(task, ensemble_accuracy))
        print_and_log('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')	
        print_and_log('Incremental Accuracy {:.2f}%'.format(all_accuracy/self.num_tasks))


    def report_scores(self):
        print_and_log('\nACCURACY MATRIX:\n {}'.format(self.acc_matrix))   
        #bwt_matrix
        for task in range(self.num_tasks-1):
            self.bwt[task] = self.acc_matrix[self.num_tasks-1, task] - self.acc_matrix[task, task]
        #inc_matrix
        final_acc = self.acc_matrix[-1]
        inc_acc = np.zeros_like(final_acc)
        for i in range(len(final_acc)):
            inc_acc[i] = final_acc[i] if i == 0 else np.mean(final_acc[:i+1])

        return final_acc, inc_acc, np.mean(self.bwt[:-1])

    def _compute_accuracy(self):
        self.net.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for batch_idx, (_, inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(torch.int64).to(self.device)
                logits = self.net(inputs)
                preds = logits.argmax(dim=1, keepdim=True)
                correct += preds.eq(targets.view_as(preds)).sum().item()
                n += len(targets)
        return np.around(100 * correct / n, decimals=2)

    def _update_used_params(self):
        current_mask = torch.load(f'./masks/{self.dataset}_task{self.cur_task}_mask.pt')
        self.used_params = {k: self.used_params.get(k, 0) + current_mask.get(k, 0) for k in set(self.used_params) | set(current_mask)}

    def _calculate_nonzeros(self):
        non_zero_count = 0
        for nonzeros in self.used_params.values():
            non_zero_count += torch.count_nonzero(nonzeros)
        if self.cur_task == self.num_tasks - 1:
            print_and_log('\nOverall Sparsity: {:.4f}.'.format(non_zero_count/self.total_params))

    def _freeze_used_params(self):
        if self.cur_task != 0:
            for name, params in self.net.named_parameters():
                if name in self.model_sparsifier.masks:
                    params.grad[self.used_params[name] != 0] = 0

    def _stack_subnets(self):
        if self.cur_task == 0:
            self.NET = copy.deepcopy(self.net)
        else:
            for (name, param), (old_name, old_param) in zip(self.net.named_parameters(), self.NET.named_parameters()):
                param.data[param == 0] = old_param.data[param == 0]
                self.NET = copy.deepcopy(self.net)
        torch.save(self.NET.state_dict(), './models/{0}_DENSE.pt'.format(self.dataset))

    def _set_mask(self, task_id):
        task_mask = torch.load(f'./masks/{self.dataset}_task{task_id}_mask.pt')
        masked_net = copy.deepcopy(self.NET)
        for n, t in masked_net.named_parameters():
            if n in task_mask: t.data = t.data * task_mask[n]
        return masked_net

    def _save_checkpoint(self):
        torch.save(self.net.state_dict(), './models/{0}.pt'.format(self.dataset))

    def _save_mask(self):
        torch.save(self.model_sparsifier.masks, './masks/{0}_task{1}_mask.pt'.format(self.dataset, self.cur_task))

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
