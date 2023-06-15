from typing import Optional, List
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainArguments(TrainingArguments):
    config_path: str = field(
        default=None, metadata={"help": "the training yaml file"}
    )

    do_reuse: bool = field(
        default=False, metadata={"help": "whether to load last checkpoint"}
    )
    metric_name: str = field(
        default="glue", metadata={"help": "whether to load last checkpoint"}
    )
    loss_name: str = field(
        default="xent", metadata={"help": "{xent: cross_entropy}"}
    )
    # is_decreased_valid_metric: bool = field(
    #     default=False
    # )  # same with `greater_is_better`
    patient_times: int = field(
        default=3,
    )
    do_grid: bool = field(
        default=False, metadata={"help": "whether to do grid search"}
    )
    crf_learning_rate: Optional[float] = field(
        default=5e-3, metadata={"help": "learning rate for crf layer"}
    )
    multi_label_threshold: Optional[List[float]] = field(
        default=None, metadata={"help": "multi label prediction threshold"}
    )
    model_save: bool = field(
        default=False, metadata={"help": "whether to save model"}
    )
    analysis: bool = field(
        default=False, metadata={"help": "whether to analysis result"}
    )
    load_model_test: bool = field(
        default=False, metadata={"help": "whether to load a trained model"}
    )
