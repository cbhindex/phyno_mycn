""" Optimizer factory for Pheno-MYCN.

Adapted and trimmed from the timm optimizer factory (Ross Wightman,
https://github.com/huggingface/pytorch-image-models, Apache-2.0). Only the
optimisers shipped with this repository are exposed:

    * sgd / momentum / nesterov
    * adam, adamw
    * radam              (vendored, see radam.py)
    * a ``lookahead_<opt>`` prefix wraps any of the above in Lookahead.

The published models were trained with ``Lookahead_radam`` (see the config
files), so that combination is the default path.
"""
import torch
from torch import optim as optim

from pheno_mycn.optimizers.radam import RAdam
from pheno_mycn.optimizers.lookahead import Lookahead


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(args, model, filter_bias_and_bn=True):
    """Build an optimiser from a config-like ``args`` object.

    ``args`` must expose ``opt`` (e.g. ``"Lookahead_radam"``), ``lr`` and
    ``weight_decay``; ``opt_eps``, ``opt_betas`` and ``momentum`` are optional.
    """
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    # The leading token (if any) selects an optimiser wrapper, e.g.
    # "lookahead_radam" -> base optimiser "radam" wrapped in Lookahead.
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    else:
        raise ValueError(f"Invalid optimizer '{args.opt}'. "
                         f"Supported: sgd, momentum, nesterov, adam, adamw, radam "
                         f"(optionally prefixed with 'lookahead_').")

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
