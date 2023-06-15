"""GPT Generation Model For FedETuning """

import copy
from abc import ABC
import torch
from utils import registry
from models.base_models import BaseModels
from transformers import AutoModelForCausalLM, GPT2LMHeadModel


@registry.register_model("seq_generation")
class ChineseSeqGeneration(BaseModels, ABC):
    def __init__(self, task_name):
        super().__init__(task_name)

        self.tokenizer = registry.get("tokenizer")
        self.auto_config = self._build_config(pad_token_id=self.tokenizer.pad_token_id)
        self.backbone = self._build_model()


    def _add_base_model(self):
        backbone = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_config.model_name_or_path),
            config=self.auto_config,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            # ignore_mismatched_sizes=self.model_config.ignore_mismatched_sizes,
        )
        return backbone

    def forward(self, inputs):
        output = self.backbone(**inputs)
        return output

    # @torch.no_grad()
    def generate(self, input_ids):
        output = self.backbone.generate(
            input_ids=input_ids,
            do_sample=False,
            max_length=self.model_config.generate_max_length,
            pad_token_id=self.auto_config.pad_token_id,  # suppress warning logging
            top_k=self.auto_config.top_k,  # default to 50
            top_p=self.auto_config.top_p,  # default to 1
            return_dict_in_generate=True,
        )
        return output
