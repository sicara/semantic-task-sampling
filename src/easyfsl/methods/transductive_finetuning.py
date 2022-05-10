import torch
import torch.nn.functional as F
from torch import Tensor, nn


from src.easyfsl.methods import AbstractClassifier
from src.easyfsl.utils import entropy


class TransductiveFinetuning(AbstractClassifier):
    """
    Implementation of Transductive Finetuning (ICLR 2020) https://arxiv.org/abs/1909.02729
    This is a transductive method.
    """

    def __init__(
        self,
        inference_steps: int = 25,
        inference_lr: float = 5e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inference_steps = inference_steps
        self.lr = inference_lr

        self.prototypes = None
        self.support_features = None
        self.support_labels = None

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
        query_features = self.backbone.forward(query_images)
        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
        for i in range(self.inference_steps):

            ce_loss = nn.functional.cross_entropy(
                self.get_logits_from_euclidean_distances_to_prototypes(
                    self.support_features
                ),
                self.support_labels,
            )
            entropy_loss = entropy(
                self.get_logits_from_euclidean_distances_to_prototypes(query_features)
            )
            loss = ce_loss + entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probs_q = self.get_logits_from_euclidean_distances_to_prototypes(
            query_features
        ).softmax(-1)

        return probs_q.detach()
