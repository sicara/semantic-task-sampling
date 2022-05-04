import torch
import torch.nn.functional as F
from torch import Tensor, nn


from easyfsl.methods import AbstractClassifier


class Finetune(AbstractClassifier):
    """
    Implementation of Finetune (or Baseline method) (ICLR 2019) https://arxiv.org/abs/1904.04232
    This is an inductive method.
    """

    def __init__(
        self,
        inference_steps: int = 10,
        inference_lr: float = 1e-3,
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

            logits_s = self.get_logits_from_cosine_distances_to_prototypes(
                self.support_features
            )
            ce = nn.functional.cross_entropy(logits_s, self.support_labels)
            optimizer.zero_grad()
            ce.backward()
            optimizer.step()

        probs_q = self.get_logits_from_cosine_distances_to_prototypes(
            query_features
        ).softmax(-1)

        return probs_q.detach()
