import copy
from abc import ABC
import torch
from torch import nn

from utils import registry
from models.base_models import BaseModels
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import RobertaForTokenClassification, trainer
from models.utils import CRF


@registry.register_model("token_classification_crf")
class TokenClassificationCRF(BaseModels, ABC):
    def __init__(self, task_name):
        super().__init__(task_name)

        self.num_labels = registry.get("num_labels")
        self.id2label = registry.get("id2label")
        self.label2id = registry.get("label2id")

        self.auto_config = self._build_config(num_labels=self.num_labels)
        self.backbone = self._build_model()

    def _add_base_model(self):
        backbone = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_config.model_name_or_path),
            config=self.auto_config,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            # ignore_mismatched_sizes=self.model_config.ignore_mismatched_sizes,
        )

        backbone.config.label2id = self.label2id
        backbone.config.id2label = self.id2label

        backbone.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # backbone.crf = CRF(num_labels=self.num_labels)

        return backbone

    def forward(self, inputs):
        outputs = self.backbone.bert(input_ids=inputs['input_ids'],
                                     attention_mask=inputs['attention_mask'],
                                     token_type_ids=inputs['token_type_ids'])
        last_encoder_layer = outputs[0]  # B, L, H
        last_encoder_layer = self.backbone.dropout(last_encoder_layer)
        emissions = self.backbone.classifier(last_encoder_layer)

        outputs = (emissions,)
        if 'labels' in inputs.keys() and inputs['labels'] is not None: # crf training
            loss = self.backbone.crf(emissions=emissions, tags=inputs['labels'], mask=inputs['attention_mask'])
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores
