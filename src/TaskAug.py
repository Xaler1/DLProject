import torch
from torch import nn, Tensor
from torch.distributions import Categorical, RelaxedOneHotCategorical
import numpy as np
from copy import deepcopy

from task_aug_operations import *
from torch.autograd import grad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Stage(nn.Module):
    def __init__(self, operations, temperature=0.05):
        super(Stage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(operations)))
        self.temperature = temperature

    def forward(self, x, y):
        if self.training:
            relaxcat = RelaxedOneHotCategorical(torch.Tensor([0.1]).to(x.device), logits=self._weights)
            wt = relaxcat.rsample()
            op_idx = wt.argmax().detach()
            op_mag = wt[op_idx] / wt[op_idx].detach()
            op_weights = torch.zeros(len(self.operations)).to(x.device)
            op_weights[op_idx] = op_mag
            return torch.stack([op_weights[i]*op(x, y) for i, op in enumerate(self.operations)]).sum(0)
        else:
            return self.operations[Categorical(logits=self._weights).sample()](x, y)

class TaskAug(nn.Module):
    def __init__(self, stage, n_operations=1, batch_first=False):
        super(TaskAug, self).__init__()
        self.batch_first = batch_first
        self.stages = nn.ModuleList([deepcopy(stage) for _ in range(n_operations)])

    def forward(self, x, y):
        if not self.batch_first:
            x = x.swapaxes(0, 1)

        x = x.swapaxes(1, 2)

        for stage in self.stages:
            x = stage(x, y)

        x = x.swapaxes(1, 2)
        if not self.batch_first:
            x = x.swapaxes(0, 1)

        return x


def full_policy(learn_mag=True, learn_prob=True, n_operations=2, num_classes=2, input_len=256, batch_first=False):
    all_ops = nn.ModuleList([
        RandTemporalWarp(initial_magnitude=1.0, learn_magnitude=learn_mag,learn_probability=learn_prob, num_classes=num_classes, input_len=input_len),
        BaselineWander(learn_magnitude=learn_mag,learn_probability=learn_prob, num_classes=num_classes),
        GaussianNoise(learn_magnitude=learn_mag,learn_probability=learn_prob, num_classes=num_classes),
        RandCrop(learn_probability=learn_prob, num_classes=num_classes),
        RandDisplacement(learn_magnitude=learn_mag,learn_probability=learn_prob, num_classes=num_classes, input_len=input_len),
        MagnitudeScale(learn_magnitude=learn_mag,learn_probability=learn_prob, num_classes=num_classes),
        NoOp(),
    ])
    return TaskAug(Stage(all_ops), n_operations=n_operations, batch_first=batch_first)

def zero_hypergrad(hyper_params):
    """
    :param get_hyper_train:
    :return:
    """
    current_index = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params

def get_hyper_train_flat(hyper_params):
    return torch.cat([p.view(-1) for p in hyper_params])

def gather_flat_grad(loss_grad):
    return torch.cat([p.reshape(-1) for p in loss_grad])

def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, list(model.parameters()), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner


def hyper_step(model, aug, hyper_params, train_loader, optimizer, val_loader, elementary_lr, neum_steps, criterion):
    zero_hypergrad(hyper_params)
    num_weights = sum(p.numel() for p in model.parameters())

    d_train_loss_d_w = torch.zeros(num_weights).to(device)
    model.train(), model.zero_grad()

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        x = aug(x, y)

        y_pred = model(x)
        train_loss = criterion(y_pred, y)
        optimizer.zero_grad()
        d_train_loss_d_w += gather_flat_grad(grad(train_loss, list(model.parameters()),
                                                  create_graph=True, allow_unused=True))
        break
    optimizer.zero_grad()

    # Initialize the preconditioner and counter
    # Compute gradients of the validation loss w.r.t. the weights/hypers
    d_val_loss_d_theta = torch.zeros(num_weights).cuda()
    model.train(), model.zero_grad()
    for batch_idx, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        val_loss = criterion(y_pred, y)
        optimizer.zero_grad()
        d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters(), retain_graph=False))
        break

    preconditioner = d_val_loss_d_theta

    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,neum_steps, model)

    indirect_grad = gather_flat_grad(
        grad(d_train_loss_d_w, hyper_params, grad_outputs=preconditioner.view(-1)))
    hypergrad = indirect_grad

    zero_hypergrad(hyper_params)
    return hypergrad


