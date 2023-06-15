"""Tuning Args"""

all_delta_config = {
    "adapter_roberta-base":
        {
            "delta_type": "adapter",
            "learning_rate": 1e-3,
            "unfrozen_modules": [
                "deltas",
                "layer_norm",
                "final_layer_norm",
                "classifier",
            ],
            "bottleneck_dim": 16,
        },
    'soft_prompt_roberta-base':
        {
            "delta_type": "soft_prompt",
            "learning_rate": 3e-2,
            "soft_token_num": 100,
            "unfrozen_modules": [
                "deltas",
                "classifier",
            ],
        },
    "lora_roberta-base":
        {
            "rte":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_rank": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 80,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "qqp":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_rank": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 25,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "mrpc":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.001,
                    "lora_alpha": 16,
                    "lora_rank": 16,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 30,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas",
                        "layer_norm"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "mnli":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_rank": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 30,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 16,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "cola":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0004,
                    "lora_alpha": 8,
                    "lora_rank": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 80,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "qnli":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0004,
                    "lora_alpha": 8,
                    "lora_rank": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 25,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                },
            "sst-2":
                {
                    "delta_type": "lora",
                    "learning_rate": 0.0005,
                    "lora_alpha": 8,
                    "lora_rank": 8,
                    "non_linearity": "gelu_new",
                    "num_train_epochs": 60,
                    "per_device_eval_batch_size": 100,
                    "per_device_train_batch_size": 32,
                    "unfrozen_modules": [
                        "classifier",
                        "deltas"
                    ],
                    "warmup_ratio": 0.06,
                    "weight_decay": 0.1,
                }

        },
    "bitfit_roberta-base":
        {
            "delta_type": "bitfit",
            "learning_rate": 3e-4,
            "unfrozen_modules": [
                "classifier",
                "deltas"
            ],
        },
    "prefix_roberta-base":
        {
            "delta_type": "prefix",
            "learning_rate": 1e-3,
            "unfrozen_modules": [
                "deltas",
                "classifier",
            ],
            "prefix_token_num": 16
        }
}

hyperparameter_grid = {

    # Hyper-parameter Setup: 6 * (1+2+2+1+3) = 54
    # adapter lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and bottleneck_dim: [16,64]
    # lora lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and lora_alpha & lora_rank from [8, 16]
    # bitfit lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    # prefix lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and prefix_token_num from [8, 16, 64]

    "mu_fine-tuning": {
        "prox_mu": [5e-5, 1e-5, 5e-4, 1e-4, 0.001, 0.002, 0.01, 0.05, 0.01, 0.5, 1.0]
    },
    "LAM_fine-tuning": {  # 11*4=44
        "learning_rate": [5e-6, 5e-5, 2e-5, 1e-5, 5e-4, 2e-4, 1e-4, 5e-3, 2e-3, 1e-3, 1e-2],
        "multi_label_threshold": [0.1, 0.3, 0.5, 0.7],
    },
    "LAM_mu_fine-tuning": {
        "learning_rate": [5e-6, 5e-5, 2e-5, 1e-5, 5e-4, 2e-4, 1e-4],
        "multi_label_threshold": [0.1, 0.3, 0.5, 0.7],
        "prox_mu": [5e-5, 1e-5, 5e-4, 1e-4, 0.001, 0.002, 0.01, 0.05, 0.01, 0.5, 1.0]
    },
    "fedopt_msgd_fine-tuning": {
        "m_t": [0.9, 0.92, 0.95, 0.98, 0.99, 0.999, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "eta": [1.0],
    },
    "LAM_fedopt_msgd_fine-tuning": {
        "learning_rate": [5e-6, 5e-5, 2e-5, 1e-5, 5e-4, 2e-4, 1e-4],
        "multi_label_threshold": [0.1, 0.3, 0.5, 0.7],
        "m_t": [0.9, 0.92, 0.95, 0.98, 0.99, 0.999, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "eta": [1.0],
    },
    "LER_fine-tuning":{
        "learning_rate": [5e-6, 2e-5, 1e-5, 5e-4, 2e-4, 1e-4, 1e-3, 5e-3],
        "crf_learning_rate": [1e-4, 5e-3, 1e-3, 5e-2, 1e-2]
    },
    "LRE_fine-tuning":{
        "learning_rate": [5e-6, 2e-5, 5e-4, 2e-4, 1e-4, 1e-3, 5e-3],
    },
    "fine-tuning": {
        "learning_rate": [5e-6, 2e-5, 1e-5, 5e-4, 2e-4, 1e-4, 1e-3, 5e-3],
        # "learning_rate": [5e-6, 5e-5, 2e-5, 1e-5, 5e-4, 2e-4, 1e-4, 1e-3, 5e-3],
        # "learning_rate": [5e-5],
    },
    "prefix": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "prefix_token_num": [8, 16, 64],
    },
    "bitfit": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    },
    "lora": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "lora_rank": [8, 16],  # so to lora_alpha
    },
    "adapter": {
        "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "bottleneck_dim": [16, 64]
    }
}

hyperparameter_grid_part = {

    # Hyper-parameter Setup: 6 * (1+2+2+1+3) = 54
    # adapter lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and bottleneck_dim: [16,64]
    # lora lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and lora_alpha & lora_rank from [8, 16]
    # bitfit lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    # prefix lr from [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] and prefix_token_num from [8, 16, 64]

    "fine-tuning": {  # 11*4=44
        "learning_rate": [1e-4, 5e-3, 2e-3, 1e-3, 1e-2],
        "multi_label_threshold": [0.1, 0.3, 0.5, 0.7],
        # "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        # "learning_rate": [5e-5],
    },
}

def get_delta_config(delta_name):
    return all_delta_config[delta_name]


def get_delta_key(delta_type):
    delta_keys = {
        "fine-tuning": "",
        "prefix": "prefix_token_num",
        "bitfit": "",
        "lora": "lora_rank",
        "adapter": "bottleneck_dim"
    }
    delta_keys_abb = {
        "fine-tuning": "",
        "prefix": "ptn",
        "bitfit": "",
        "lora": "la",
        "adapter": "dim"
    }
    return delta_keys[delta_type], delta_keys_abb[delta_type]
