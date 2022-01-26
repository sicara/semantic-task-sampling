"""
This code is extracted from https://github.com/yhu01/PT-MAP/blob/master/test_standard.py
We only added minimal fixes to make it work in our code base.
We can't promise anything about the quality of the code in this file.
"""


import torch

from torch import nn, Tensor

# ========================================
#      loading datas
from easyfsl.methods import AbstractMetaLearner


def centerDatas(datas, n_lsamples):
    support_means = datas[:, :n_lsamples].mean(1, keepdim=True)
    query_means = datas[:, n_lsamples:].mean(1, keepdim=True)
    support_norm = torch.norm(datas[:, :n_lsamples], 2, 2)[:, :, None]
    query_norm = torch.norm(datas[:, n_lsamples:], 2, 2)[:, :, None]

    datas_out = torch.zeros_like(datas)
    datas_out[:, :n_lsamples] = (datas[:, :n_lsamples] - support_means) / support_norm
    datas_out[:, n_lsamples:] = (datas[:, n_lsamples:] - query_means) / query_norm

    return datas_out


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


# ---------  GaussianModel
class GaussianModel:
    def __init__(
        self,
        n_way,
        lam,
        ndatas,
        n_runs,
        n_shot,
        n_query,
        n_nfeat,
        n_lsamples,
        n_usamples,
    ):
        self.n_way = n_way
        self.mus = None  # shape [n_runs][n_way][n_nfeat]
        self.lam = lam
        self.n_runs = n_runs
        self.n_query = n_query
        self.n_lsamples = n_lsamples
        self.n_usamples = n_usamples
        self.mus = ndatas.reshape(n_runs, n_shot + n_query, n_way, n_nfeat)[
            :,
            :n_shot,
        ].mean(1)

    def cuda(self):
        # Inplace
        self.mus = self.mus.cuda()

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):

        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(-self.lam * M)
        P = P / P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P = P * (r / u).view((n_runs, -1, 1))
            P = P * (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    def getProbas(self, ndatas, labels):
        # compute squared dist to centroids [n_runs][n_samples][n_way]
        dist = (ndatas.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(self.n_runs, self.n_usamples)
        c = torch.ones(self.n_runs, self.n_way) * self.n_query

        p_xj_test, _ = self.compute_optimal_transport(
            dist[:, self.n_lsamples :], r, c, epsilon=1e-6
        )
        p_xj[:, self.n_lsamples :] = p_xj_test

        p_xj[:, : self.n_lsamples] = p_xj[:, : self.n_lsamples].scatter(
            2, labels[:, : self.n_lsamples].unsqueeze(2), 1
        )

        return p_xj

    def estimateFromMask(self, mask, ndatas):

        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus


class PT_MAP(AbstractMetaLearner):
    def __init__(self, model_func, power_transform=True):
        super().__init__(model_func)
        self.loss_fn = nn.NLLLoss()
        self.power_transform = power_transform

    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Overrides process_support_set of AbstractMetaLearner.
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        self.store_features_labels_and_prototypes(support_images, support_labels)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:

        n_way = len(self.prototypes)
        n_shot = len(self.support_labels) // n_way
        n_query = len(query_images) // n_way
        n_runs = 1
        n_lsamples = n_way * n_shot
        n_usamples = n_way * n_query
        n_samples = n_lsamples + n_usamples
        self.n_lsamples = n_lsamples
        self.n_runs = n_runs

        query_features = self.backbone.forward(query_images)
        label_mapping = (
            self.support_labels.view(n_way, n_shot).permute(1, 0).sort()[1][0]
        )

        support_mapping = torch.cat([label_mapping * n_shot + i for i in range(n_shot)])
        query_mapping = torch.cat([label_mapping * n_query + i for i in range(n_query)])

        ndatas = torch.cat(
            (self.support_features[support_mapping], query_features[query_mapping]),
            dim=0,
        ).unsqueeze(0)
        labels = (
            torch.arange(n_way)
            .view(1, 1, n_way)
            .expand(n_runs, n_shot + n_query, 5)
            .clone()
            .view(n_runs, n_samples)
        )

        if self.power_transform:
            # Power transform
            beta = 0.5
            ndatas[:,] = torch.pow(
                ndatas[
                    :,
                ]
                + 1e-6,
                beta,
            )

        ndatas = QRreduction(ndatas)  # Now ndatas has shape (1, n_samples, n_samples)
        n_nfeat = ndatas.size(2)

        ndatas = scaleEachUnitaryDatas(ndatas)

        # trans-mean-sub

        ndatas = centerDatas(ndatas, n_lsamples)

        # switch to cuda
        ndatas = ndatas.cuda()
        labels = labels.cuda()

        # MAP
        lam = 10
        model = GaussianModel(
            n_way,
            lam,
            ndatas,
            n_runs,
            n_shot,
            n_query,
            n_nfeat,
            n_lsamples,
            n_usamples,
        )

        self.alpha = 0.2

        self.ndatas = ndatas
        self.labels = labels

        probas = self.loop(model, n_epochs=20)

        # TODO remettre les labels dans le sens originel

        return probas.squeeze(0)[n_lsamples:][query_mapping.sort()[1]]

    def performEpoch(self, model):
        p_xj = model.getProbas(self.ndatas, self.labels)
        self.probas = p_xj

        m_estimates = model.estimateFromMask(self.probas, self.ndatas)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

    def loop(self, model, n_epochs=20):
        self.probas = model.getProbas(self.ndatas, self.labels)

        for epoch in range(1, n_epochs + 1):
            self.performEpoch(model)

        # get final accuracy and return it
        op_xj = model.getProbas(self.ndatas, self.labels)
        return op_xj
