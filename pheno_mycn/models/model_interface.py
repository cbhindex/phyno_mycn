"""
PyTorch-Lightning training/validation/test interface for Pheno-MYCN.

``ModelInterface`` is the LightningModule that wraps any of the bag-level models
(Pheno-MYCN / CLAM-SB, TransMIL, MILNet_multi), drives the training and
evaluation loops, and — for Pheno-MYCN — initialises and supervises the
auxiliary GMM phenotype branch.

Key responsibilities:
  * dynamic model instantiation from the config (``Model.name``);
  * ProtoDiv-based initialisation of the GMM means/covariances at train start
    from MYCN-amplified tiles (``on_train_start``);
  * the composite objective: classification loss + 1e-3 * auxiliary GMM energy;
  * AUROC / accuracy / F1 / precision / recall logging per fold.

The number of GMM components K is taken from the config (``Model.l``); the
manuscript fixes K = 6.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:                     Dr Olga Fourkioti   (https://github.com/olgarithmics)
Code review & refactoring:  Dr Binghao Chai     (https://bhchai.com/, https://github.com/cbhindex)

License: GPL-3.0 (see the LICENSE file at the repository root).
Portions adapted from CLAM (Mahmood Lab, GPL-3.0).
"""

import inspect
import importlib
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from pheno_mycn.optimizers import create_optimizer
from pheno_mycn.losses import create_loss


class ModelInterface(pl.LightningModule):

    # ---- init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()

        self.loss = create_loss(loss)
        self.optimizer = optimizer

        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.lr = self.optimizer.lr

        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # ---- metrics
        if self.n_classes > 2:
            self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average='macro', task="multiclass")
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes=self.n_classes, task="multiclass", average='micro'),
                torchmetrics.F1Score(num_classes=self.n_classes, task="multiclass", average='macro'),
                torchmetrics.Recall(average='macro', task="multiclass", num_classes=self.n_classes),
                torchmetrics.Precision(average='macro', task="multiclass", num_classes=self.n_classes),
                torchmetrics.Specificity(average='macro', task="multiclass", num_classes=self.n_classes),
            ])
        else:
            if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
                self.AUROC = torchmetrics.AUROC(task="binary")
                metrics = torchmetrics.MetricCollection({
                    'accuracy': torchmetrics.Accuracy(task="binary", average='micro'),
                    'f1_score': torchmetrics.F1Score(task="binary", average='macro'),
                    'recall': torchmetrics.Recall(task="binary", average='macro'),
                    'precision': torchmetrics.Precision(task="binary", average='macro'),
                })
            else:
                self.AUROC = torchmetrics.AUROC(num_classes=2, average='macro', task="multiclass")
                metrics = torchmetrics.MetricCollection({
                    'accuracy': torchmetrics.Accuracy(num_classes=2, average='micro', task="multiclass"),
                    'f1_score': torchmetrics.F1Score(num_classes=2, average='macro', task="multiclass"),
                    'recall': torchmetrics.Recall(average='macro', num_classes=2, task="multiclass"),
                    'precision': torchmetrics.Precision(average='macro', num_classes=2, task="multiclass"),
                })

        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Number of GMM components K (manuscript: 6). Read from the config.
        self.l = model.l

        self.clustering_method = 'ProtoDiv'
        self.proto_method = 'mean'
        self.pheno_cut_method = 'quantile'
        self.iter_fine_tuning = 20

        self.shuffle = kargs['data'].data_shuffle
        self.count = 0

    @staticmethod
    def uniform_assign(N, num_label):
        L = torch.randperm(N) % num_label
        rlab = torch.randperm(num_label)
        res = rlab[L]
        return res

    @staticmethod
    def mean_by_label(samples, labels):
        """select mean(samples), count() from samples group by labels, ordered by labels ASC."""
        weight = torch.zeros(labels.max() + 1, samples.shape[0]).to(samples.device)  # #class, N
        weight[labels, torch.arange(samples.shape[0])] = 1
        label_count = weight.sum(dim=1)
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalisation
        mean = torch.mm(weight, samples)  # #class, F
        index = torch.arange(mean.shape[0])[label_count > 0]
        return mean[index], label_count[index]

    def get_phenotype_clusters(self, bag, **kws):
        if self.clustering_method == 'ProtoDiv':
            assert 'ptype' in kws
            centroids, label_phe = self.protodiv_clustering(bag, ptype=kws['ptype'])
        else:
            clusters = None
        return centroids, label_phe

    def protodiv_clustering(self, bag, ptype=None, metric='cosine'):
        # calculate distances
        dis, limits = self.protodiv_measure_distance(bag, ptype=ptype)  # [N, ], tuple

        if self.pheno_cut_method == 'quantile':
            data = dis.cpu().numpy()
            bins = np.quantile(data, [i / self.l for i in range(self.l + 1)])
            bins[0], bins[-1] = bins[0] - 1e-5, bins[-1] + 1e-5
            label_phe = np.digitize(data, bins) - 1
            label_phe = torch.LongTensor(label_phe).to(dis.device)
        else:
            pass

        if self.iter_fine_tuning > 0:
            ind_cluster = label_phe
            for i in range(self.iter_fine_tuning):
                centroids, _ = self.mean_by_label(bag, ind_cluster)
                if metric == 'cosine':
                    norm_centroids = F.normalize(centroids, p=2, dim=-1)  # [l, d]
                    norm_X = F.normalize(bag, p=2, dim=-1)
                    dis = torch.mm(norm_X, norm_centroids.T)  # [N, d] x [d, l] -> [N, l]
                else:
                    raise NotImplementedError("cannot recognize {}".format(metric))
                _, new_ind = torch.max(dis, dim=1)
                ind_cluster = new_ind
            label_phe = ind_cluster
        return centroids, label_phe

    def compute_covariances(self, features, labels, centroids, epsilon=1e-6):
        """Compute the diagonal covariance for each cluster.

        Args:
            features: Tensor [N, D] (N samples, D feature dimensions).
            labels: Tensor [N] (cluster assignment per sample).
            centroids: Tensor [K, D] (cluster mean vectors).
            epsilon: small regularisation term to prevent zero variance.

        Returns:
            variances: Tensor [K, D], per-cluster diagonal variances.
        """
        variances = []
        for k in range(self.l):
            cluster_points = features[labels == k]  # points in cluster k
            if len(cluster_points) > 1:
                variance = cluster_points.var(dim=0, unbiased=True)
            else:
                variance = torch.ones_like(centroids[k])  # default variance for empty clusters
            variances.append(variance + epsilon)  # add epsilon for stability
        return torch.stack(variances)

    def protodiv_measure_distance(self, X, ptype=None, metric='cosine'):
        if ptype is not None:
            ptype = ptype
        elif self.proto_method == 'mean':
            ptype = torch.mean(X, dim=0)
        elif self.proto_method == 'max':
            ptype, _ = torch.max(X, dim=0)

        norm_ptype = F.normalize(ptype, p=2, dim=-1)
        assert X.shape[-1] == norm_ptype.shape[-1]

        if metric == 'cosine':
            norm_X = F.normalize(X, p=2, dim=-1)
            dis = torch.mm(norm_X, norm_ptype.view(-1, 1)).squeeze()
            limits = (-1, 1)
        else:
            raise NotImplementedError("cannot recognize {}".format(metric))
        return dis, limits

    def on_train_start(self):
        """Initialise the GMM means/covariances from MYCN-amplified training tiles."""
        n_patches_per_batch = 500
        class_proto = []
        train_dataloader = self.trainer.datamodule.train_dataloader()
        for batch in train_dataloader:
            data, label, slide_id = batch
            if label.item() == 1:
                n_samples = int(n_patches_per_batch)
                if len(data.shape) > 2:
                    data = data.squeeze(0)
                indices = torch.randperm(len(data))[:n_samples]
                with torch.no_grad():
                    out = data[indices].reshape(-1, data.shape[-1])
                class_proto.append(out)
            else:
                continue

        class_proto = np.vstack(class_proto)
        class_proto = torch.tensor(class_proto, dtype=torch.float32)
        centroids, label_phe = self.get_phenotype_clusters(class_proto, ptype=None)

        centroids = torch.tensor(centroids, dtype=torch.float32)
        covariances = self.compute_covariances(class_proto, label_phe, centroids)
        covariances = covariances.to(dtype=torch.float32, device='cuda')

        self.model.ad_layer.V_.data = torch.log(torch.exp(covariances) - 1)
        self.model.ad_layer.mu.data = centroids.cuda()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        data, label, slide_id = batch

        results_dict = self.model(data, label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        anomaly_loss = results_dict['anomaly_loss']
        inst_logit = None

        if 'max_prediction' in results_dict.keys():
            inst_logit = results_dict['max_prediction']

        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            label = label.view_as(logits).float()
            loss_total = self.loss(logits, label)
        else:
            loss_total = self.loss(logits, label)

        if inst_logit is not None:
            loss_max = self.loss(inst_logit, label)
            loss_total = 0.5 * loss_total + 0.5 * loss_max

        loss = loss_total + 0.001 * anomaly_loss

        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss, 'y_hat': Y_hat, 'y_true': Y}

    def training_epoch_end(self, outputs):
        if self.trainer.is_global_zero:
            print(">>> training_epoch_end called")

        correct_counts = [0 for _ in range(self.n_classes)]
        total_counts = [0 for _ in range(self.n_classes)]
        for out in outputs:
            y_pred = out['y_hat']
            y_true = out['y_true']
            total_counts[y_true] += 1
            correct_counts[y_true] += int(y_pred == y_true)

        if self.trainer.is_global_zero:
            for c in range(self.n_classes):
                count = total_counts[c]
                correct = correct_counts[c]
                acc = None if count == 0 else correct / count
                print(f'class {c}: acc {acc}, correct {correct}/{count}')

    def validation_step(self, batch, batch_idx):
        data, label, slide_id = batch

        results_dict = self.model(data, label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        Y_prob = results_dict['Y_prob']
        anomaly_loss = results_dict['anomaly_loss']

        inst_logit = None
        if 'max_prediction' in results_dict.keys():
            inst_logit = results_dict['max_prediction']

        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            label = label.view_as(logits).float()
            loss_total = self.loss(logits, label)
        else:
            loss_total = self.loss(logits, label)

        if inst_logit is not None:
            loss_max = self.loss(inst_logit, label)
            loss_total = 0.5 * loss_total + 0.5 * loss_max
        loss = loss_total + 0.001 * anomaly_loss

        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label, 'loss': loss}

    def validation_epoch_end(self, val_step_outputs):
        if self.trainer.is_global_zero:
            print(">>> validation_epoch_end called")

        logits = torch.cat([x['logits'] for x in val_step_outputs], dim=0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim=0)
        loss = torch.stack([x['loss'] for x in val_step_outputs], dim=0)

        # Log metrics
        self.log('val_loss', torch.mean(loss), prog_bar=False, on_epoch=True, logger=True)
        self.log('val_auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()), on_epoch=True, logger=True)

        # Accuracy logging by class
        correct_counts = [0 for _ in range(self.n_classes)]
        total_counts = [0 for _ in range(self.n_classes)]
        for x in val_step_outputs:
            y_pred = int(x['Y_hat'])
            y_true = int(x['label'])
            total_counts[y_true] += 1
            correct_counts[y_true] += int(y_pred == y_true)

        if self.trainer.is_global_zero:
            for c in range(self.n_classes):
                count = total_counts[c]
                correct = correct_counts[c]
                acc = None if count == 0 else correct / count
                print(f'class {c}: acc {acc}, correct {correct}/{count}')

        # Optional reshuffle logic
        if self.shuffle:
            self.count += 1
            random.seed(self.count * 50)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=self.lr / 50, verbose=True,
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def test_step(self, batch, batch_idx):
        data, label, slide_id = batch
        results_dict = self.model(data, label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # ---- acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        # To export per-tile GMM responsibilities / attention / projected features
        # for the downstream phenotype analyses, save the relevant entries of
        # ``results_dict`` here (e.g. ``gmm_sores``, ``scores``, ``feats``). See
        # the experiments/ scripts for the expected on-disk layout.

        return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim=0)

        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()

        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            acc = None if count == 0 else float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')

    def load_model(self):
        """Dynamically import and instantiate the model named in the config.

        The module file name must match the class name, e.g. ``Model.name:
        CLAM_SB`` loads ``pheno_mycn.models.CLAM_SB.CLAM_SB``.
        """
        name = self.hparams.model.name
        try:
            Model = getattr(importlib.import_module(f'pheno_mycn.models.{name}'), name)
        except Exception:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """Instantiate ``Model`` using matching parameters from ``self.hparams.model``.

        Any keyword in ``other_args`` overrides the corresponding config value.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)
