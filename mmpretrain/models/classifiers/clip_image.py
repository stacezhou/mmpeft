from .clip.model import CLIP as originCLIP
from .clip import tokenize

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier
from ..heads.cls_head import ClsHead
class LossHeadMixin(ClsHead):
    def __init__(self, *args, **kwargs):
        super(ClsHead, self).__init__()
        

@MODELS.register_module()
class CLIPClassifier(originCLIP, BaseClassifier, LossHeadMixin):
    def __init__(self, 
            *args,
            loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk: Union[int, Tuple[int]] = (1, ),
            cal_acc: bool = False,
            **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc
        self.topk = topk

    @property
    def prompts(self):
        return self._prompts
    
    def set_prompts(self, prompts):
        self.cache_text_feats = None
        self._prompts = prompts

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, stage='neck'):
        return self.encode_image(inputs)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        image_features = self.extract_feat(inputs)
        text_features = self.extract_prompts_feat()
        logits_per_image = self.compute_logits(image_features, text_features)

        losses = self._get_loss(logits_per_image, data_samples)
        return losses

    def compute_logits(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        image_features = self.extract_feat(inputs)
        text_features = self.extract_prompts_feat()
        logits_per_image = self.compute_logits(image_features, text_features)
        pred_scores = logits_per_image.softmax(dim=-1)
        return self._get_predictions(pred_scores, data_samples)

    def extract_prompts_feat(self):
        if self.cache_text_feats is not None and True: #todo cache text feat
            return self.cache_text_feats

        with torch.no_grad(): #todo
            flat_prompts = [prompt for prompts in self.prompts for prompt in prompts]
            num_classes = len(self.prompts)
            num_templates = len(self.prompts[0])
            text_ids = tokenize(flat_prompts).to(self.logit_scale.device)
            all_text_feats = []
            for i in range(text_ids.shape[0] // 1000 + 1):
                text_feat = self.encode_text(text_ids[i*1000:(i+1)*1000])
                all_text_feats.append(text_feat)
            text_feats = torch.cat(all_text_feats, dim=0)
            text_feats = text_feats.reshape(num_classes, num_templates, -1).mean(dim=1)

            self.cache_text_feats = text_feats.detach()

        return text_feats