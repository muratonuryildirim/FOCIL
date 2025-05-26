import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
from utils import CosineDecay

class ModelSparsifier:
    def __init__(self, net=None, sparsity=None, sparse_init=None, death_mode=None, death_rate=None, growth_mode=None, 
                 update_frequency=None, optimizer=None, train_loader=None, device=None, epochs=None):
        
        self.density = 1 - sparsity
        self.sparse_init = sparse_init
        self.death_mode = death_mode
        self.death_rate = death_rate
        self.growth_mode = growth_mode
        self.update_frequency = update_frequency
        self.optimizer = optimizer
        self.train_loader = train_loader
        #self.epochs = epochs
        self.device = device

        self.masks = {}
        self.modules = [net]
        self.names = []

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.steps = 0

        #GMP
        #if self.sparse_init == 'gmp':
        #    self.death_start = epochs/2
        #    self.death_end = epochs - (epochs/10)

        #DST
        if self.update_frequency is not None:
            self.death_rate_decay = self._drate_scheduler()

        self._init()

    def apply(self):
        self._apply_mask()

        #if self.sparse_init == 'gmp':
        #    self._truncate_weights_GMP(epoch)

        #elif self.update_frequency is not None:
        #    self.death_rate_decay.step()
        #    self.death_rate = self.death_rate_decay.get_dr()
        #    self.steps += 1
            
        #    if self.steps % self.update_frequency == 0:
        #        self._truncate_weights_DST()
        #        _, _ = self.fired_masks_update()

    def _init(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                self.names.append(name)
                self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)
            self._discard_weight_partial_name('bias')
            self._discard_type(nn.BatchNorm2d)
            self._discard_type(nn.BatchNorm1d)
        
        if self.sparse_init == 'uniform':
            self._uniform()
        elif self.sparse_init == 'erk':
            self._erk()
        elif self.sparse_init == 'snip':
            self._snip()
        elif self.sparse_init == 'grasp':
            self._grasp()
        elif self.sparse_init == 'uniform_plus':
            self._uniform_plus()
        elif self.sparse_init == 'erk_plus':
            self._erk_plus()
        elif self.sparse_init == 'gmp':
            self._gmp()
        elif self.sparse_init == 'global_magnitude':
            self._global_magnitude()

        self._apply_mask()
        self.fired_masks = copy.deepcopy(self.masks)

    def _truncate_weights_DST(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                if self.death_mode == 'magnitude':
                    new_mask = self._magnitude_death(mask, weight, name)
                elif self.death_mode == 'set':
                    new_mask = self._magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == 'taylor_fo':
                    new_mask = self._taylor_FO(mask, weight, name)
                elif self.death_mode == 'mest':
                    new_mask = self._mest_death(mask, weight, name)
                elif self.death_mode == 'sensitivity':
                    new_mask = self._sensitivity_death(mask, weight, name)
                elif self.death_mode == 'r_sensitivity':
                    new_mask = self._reciprocal_sensitivity_death(mask, weight, name)
                elif self.death_mode == 'snip':
                    new_mask = self._snip_death(mask, weight, name)

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self._random_growth(name, new_mask, weight)
                if self.growth_mode == 'unfired':
                    new_mask = self._unfired_growth(name, new_mask, weight)
                elif self.growth_mode == 'gradient':
                    new_mask = self._gradient_growth(name, new_mask, weight)
                elif self.growth_mode == 'momentum':
                    new_mask = self._momentum_growth(name, new_mask, weight)

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

        self._apply_mask()

    def _truncate_weights_GMP(self, epoch):
        death_rate = 1 - self.density
        death_interval = self.death_end - self.death_start + 1
        
        if epoch >= self.init_death_epoch and epoch <= self.death_end:
            death_decay = (1 - ((epoch - self.death_start) / death_interval)) ** 3
            curr_death_rate = death_rate - (death_rate * death_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    score = torch.abs(weight.data)
                    x, idx = torch.sort(score.view(-1))
                    p = int(curr_death_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self._apply_mask()
        # for sanity check can be removed    
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density,
                                                                                            sparse_size / total_size,
                                                                                            epoch))
        
    '''
                    DEATH
    '''
    def _threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def _magnitude_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        if num_remove == 0.0: return weight.data != 0.0
        
        score = torch.abs(weight.data)
        x, idx = torch.sort(score.view(-1))
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask
    
    def _taylor_FO(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        score = (weight.data * weight.grad).pow(2)
        x, idx = torch.sort(score.flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask
    
    def _mest_death(self, mask, weight, name, gamma=1.0):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        score = abs(weight.data) + gamma * abs(weight.grad)
        x, idx = torch.sort(score.flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask
    
    def _sensitivity_death(self, mask, weight, name, epsilon=1e-8):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        score = (abs(weight.grad) / (abs(weight.data) + epsilon)) + 1
        x, idx = torch.sort(score.flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask
    
    def _reciprocal_sensitivity_death(self, mask, weight, name, epsilon=1e-8):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        score = (abs(weight.data) / (abs(weight.grad) + epsilon)) + 1
        x, idx = torch.sort(score.flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask
    
    def _magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        score = weight[weight > 0.0].data
        x, idx = torch.sort(score.view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]: k = x.shape[0]
        threshold_magnitude = x[k - 1].item()

        score = weight[weight < 0.0].data
        x, idx = torch.sort(score.view(-1))
        threshold_negativity = x[k - 1].item()

        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)
        new_mask = pos_mask | neg_mask

        return new_mask
    
    def _snip_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        score = abs(weight.data * weight.grad)
        x, idx = torch.sort(score.flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    '''
                    GROWTH
    '''
    def _random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth / n)
        new_weights = torch.rand(new_mask.shape).to(self.device) < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_ != 0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_
    
    def _unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name] == 0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name] == 0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth - num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).to(self.device) < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def _gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self._get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask
    
    def _momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self._get_momentum_for_weights(weight)
        grad = grad * (new_mask == 0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    '''
                INIT
    '''
    def _gmp(self):
        #print('initialize by gmp')
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).to(self.device)

    def _erk(self):
        #print('initialize by erk')
        erk_power_scale=1.0
        is_epsilon_valid = False

        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - self.density)
                n_ones = n_param * self.density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (np.sum(mask.shape) / np.prod(mask.shape)) ** erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        #print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.masks.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one

            self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.to(self.device)

    def _uniform(self):
        #print('initialize by uniform')
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name][:] = (torch.rand(weight.shape) < self.density).float().data.to(self.device)  # lsw

    def _snip(self):
        #print('initialize by snip')
        # Grab a single batch from the training dataset
        inputs, targets = next(iter(self.train_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs.requires_grad = True

        # Let's create a fresh copy of the network so that we're not worried about
        # affecting the actual training-phase
        for module in self.modules:
            #net = copy.deepcopy(module).to(self.device)
            module.to(self.device)
            # Compute gradients (but don't apply them)
            module.zero_grad()
            outputs = module(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            grads_abs = []
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                grads_abs.append(torch.abs(weight.grad))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
            #norm_factor = torch.sum(all_scores)
            #all_scores.div_(norm_factor)

            num_params_to_keep = int(len(all_scores) * self.density)
            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            snip_masks = []
            for g in grads_abs:
                snip_masks.append(((g) >= acceptable_score).float())
            #print(torch.sum(torch.cat([torch.flatten(x == 1) for x in snip_masks])))

            module.zero_grad()

            # re-sample mask positions
            for snip_mask, name in zip(snip_masks, self.masks):
                assert (snip_mask.shape == self.masks[name].shape)
                self.masks[name][:] = snip_mask
                #self.masks[name][:] = (torch.rand(self.masks[name].shape) < self.density).float().data.to(self.device)
       
    def _uniform_plus(self):
        #print('initialize by uniform+')
        total_params = 0
        for name, weight in self.masks.items():
            total_params += weight.numel()
        total_sparse_params = total_params * self.density

        # remove the first layer
        total_sparse_params = total_sparse_params - self.masks['conv.weight'].numel()
        self.masks.pop('conv.weight')

        if self.density < 0.2:
            total_sparse_params = total_sparse_params - self.masks['fc.weight'].numel() * 0.2
            self.density = float(total_sparse_params / total_params)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    if name != 'fc.weight':
                        self.masks[name][:] = (torch.rand(weight.shape) < self.density).float().data.cuda()
                    else:
                        self.masks[name][:] = (torch.rand(weight.shape) < 0.2).float().data.cuda()
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < self.density).float().data.cuda()

    def _erk_plus(self):
        #print('initialize by ERK_plus')
        erk_power_scale = 1.0
        total_params = 0
        self.baseline_nonzero = 0
        for name, weight in self.masks.items():
            total_params += weight.numel()
            self.baseline_nonzero += weight.numel() * density

        for name in self.masks.copy():
            if 'fc.weight' in name:
                total_params = total_params - self.masks[name].numel()
                density = (self.baseline_nonzero - self.masks[name].numel() * self.fc_density) / total_params
                self.masks.pop(name)

        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    if len(mask.shape) !=2 :
                        raw_probabilities[name] = (
                                                    np.sum(mask.shape) / np.prod(mask.shape)
                                            ) ** erk_power_scale
                    else:
                        raw_probabilities[name] = (
                                                        np.sum(mask.shape) / np.prod(mask.shape)
                                                ) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        #print(f"Sparsity of var:{mask_name} had to be set to 0.") # can be removed   
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.masks.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            #print(f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}") # can be removed   
            self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += density_dict[name] * mask.numel()

        for name, weight in self.module.named_parameters():
            if 'fc.weight' in name:
                self.masks[name] = (torch.rand(weight.shape) < self.fc_density).float().data.cuda()
                total_nonzero += self.fc_density * weight.numel()
                total_params += weight.numel()
                #print(f"layer: {name}, shape: {self.masks[name].shape}, density: {self.fc_density}")
        #print(f"Overall sparsity {total_nonzero / total_params}") # can be removed   

    def _global_magnitude(self):
        #print('initialize by global magnitude')
        weight_abs = []
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                weight_abs.append(torch.abs(weight))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(len(all_scores) * self.density)

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

    '''
                UTILITY
    '''
    def _apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]

    def _discard_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                removed.add(name)
                self.masks.pop(name)
        
        ##is this necessary? 
        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def _discard_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self._discard_weight(name)

    def _discard_weight(self, name):
        if name in self.masks:
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            self.masks.pop(name + '.weight')

    def _get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def _get_momentum_for_weights(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad
    
    #def _drate_scheduler(self):
    #    death_rate_scheduler = CosineDecay(self.death_rate, len(self.train_loader) * self.epochs)
    #    return death_rate_scheduler

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item()) / float(
                    self.fired_masks[name].numel())
                #print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights / ntotal_weights
        #print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights
    
def _snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def _snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

