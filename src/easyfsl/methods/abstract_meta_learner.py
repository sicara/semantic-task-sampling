from abc import abstractmethod
import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.easyfsl.utils import (
    compute_backbone_output_shape,
    get_task_perf,
    compute_prototypes,
)


class AbstractClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(
        self,
        backbone: nn.Module,
        tensorboard_writer: SummaryWriter = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.backbone = backbone
        self.backbone_output_shape = compute_backbone_output_shape(backbone)
        self.feature_dimension = self.backbone_output_shape[0]
        self.loss_function = nn.CrossEntropyLoss()

        self.best_validation_accuracy = 0.0
        self.best_model_state = None

        self.training_tasks_record = []
        self.training_confusion_matrix = None

        self.tensorboard_writer = tensorboard_writer
        if not tensorboard_writer:
            logger.warning(
                "No tensorboard writer specified. Training curves won't be logged."
            )

        self.device = torch.device(device=device)

    # pylint: disable=all
    @abstractmethod
    def forward(
        self,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict classification labels.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    @abstractmethod
    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using
        a forward call

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a process_support_set method."
        )

    # pylint: enable=all

    def infer_on_one_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the labels of query images given a few labelled support examples.
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
            query_images: images of the query set
        Returns:
            classification scores of shape (number_of_query_images, n_way)
        """

        self.process_support_set(
            support_images.to(self.device), support_labels.to(self.device)
        )

        return self(query_images.to(self.device)).detach()

    def evaluate(self, data_loader: DataLoader) -> pd.DataFrame:
        """
        Evaluate the model on few-shot classification tasks
        Args:
            data_loader: loads data in the shape of few-shot classification tasks
        Returns:
            average classification accuracy
        """
        list_of_task_perfs = []

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph
        self.eval()
        # with torch.no_grad():
        with tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Evaluation"
        ) as tqdm_eval:
            for task_id, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                true_class_ids,
            ) in tqdm_eval:
                predicted_scores = self.infer_on_one_task(
                    support_images, support_labels, query_images
                )
                list_of_task_perfs.append(
                    get_task_perf(
                        task_id, predicted_scores, query_labels, true_class_ids
                    ).assign(
                        task_loss=self.compute_loss(
                            predicted_scores, query_labels.to(self.device)
                        ).item()
                    )
                )

        return pd.concat(list_of_task_perfs, ignore_index=True)

    def compute_loss(
        self, classification_scores: torch.Tensor, query_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the method's criterion to compute the loss between predicted classification scores,
        and query labels.
        We do this in a separate function because some few-shot learning algorithms don't apply
        the loss function directly to classification scores and query labels. For instance, Relation
        Networks use Mean Square Error, so query labels need to be put in the one hot encoding.
        Args:
            classification_scores: predicted classification scores of shape (n_query, n_classes)
            query_labels: ground truth labels. 1-dim tensor of length n_query

        Returns:
            loss
        """
        return self.loss_function(classification_scores, query_labels)

    def get_logits_from_euclidean_distances_to_prototypes(self, samples):
        return -torch.cdist(samples, self.prototypes)

    def get_logits_from_cosine_distances_to_prototypes(self, samples):
        return F.normalize(samples, dim=1) @ F.normalize(self.prototypes, dim=1).T

    def store_features_labels_and_prototypes(
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
        self.support_labels = support_labels
        self.support_features = self.backbone.forward(support_images)
        self.prototypes = compute_prototypes(self.support_features, support_labels)
