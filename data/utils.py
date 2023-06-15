"""Data Process Utils"""
import numpy as np
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED, `0` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None


def tokenize_and_align_labels(seq_tokens, seq_labels, tokenizer,
                              label_to_id, b_to_i_label, max_seq_length,
                              padding=False, label_all_tokens=False, crf_process=False):
    tokenized_inputs = tokenizer(
        seq_tokens,
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(seq_labels):
        mask = tokenized_inputs['attention_mask'][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx, mask_idx in zip(word_ids, mask):
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                if not crf_process:
                    label_ids.append(-100)
                elif mask_idx:
                    label_ids.append(label_to_id['O'])  # special token but mask=1
                else:
                    label_ids.append(label_to_id['X'])  # other special token

            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    if not crf_process:
                        label_ids.append(-100)
                    else:
                        label_ids.append(label_to_id['X'])

            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["label"] = labels
    return tokenized_inputs


def conll_convert_examples_to_features(examples, tokenizer, max_length, label_list,
                                       output_mode, label_all_tokens=False, padding=True):
    seq_tokens = [example["tokens"] for example in examples]
    seq_labels = [example["labels"] for example in examples]

    label_to_id = {l: i for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    tokenized_inputs = tokenize_and_align_labels(
        seq_tokens, seq_labels, tokenizer,
        label_to_id, b_to_i_label, max_length,
        padding=padding, label_all_tokens=label_all_tokens)

    keys = ['input_ids', 'attention_mask', 'token_type_ids', 'label']
    features = []
    for i in range(len(tokenized_inputs["input_ids"])):
        temp = {}
        for key in keys:
            if key in tokenized_inputs:
                temp[key] = tokenized_inputs[key][i]
        features.append(InputFeatures(**temp))

    return features


def action_legal_examples_to_features(examples, tokenizer, max_length, label_list,
                                      output_mode, label_all_tokens=False, padding=True):
    if max_length is None:
        max_length = tokenizer.model_max_length

    label_to_id = {l: i for i, l in enumerate(label_list)}
    print(label_to_id)

    if "token_classification" in output_mode:
        b_to_i_label = []

        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        tokens = [example["tokens"] for example in examples]
        labels = [example["label"] for example in examples]
        tokenized_inputs = tokenize_and_align_labels(
            tokens, labels, tokenizer,
            label_to_id, b_to_i_label, max_length,
            padding=padding, label_all_tokens=label_all_tokens, crf_process='crf' in output_mode)

        keys = ['input_ids', 'attention_mask', 'token_type_ids', 'label']
        features = []
        for i in range(len(tokenized_inputs["input_ids"])):
            temp = {}
            for key in keys:
                if key in tokenized_inputs:
                    temp[key] = tokenized_inputs[key][i]
            features.append(InputFeatures(**temp))

    else:
        batch_encoding = tokenizer(
            [(example['text_a'], example['text_b']) if example['text_b'] != "" else example['text_a'] for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []

        # label process
        if output_mode == 'seq_generation':
            pass
        elif 'multi' in output_mode:
            labels = []
            for example in examples:
                label_indices = [label_to_id[l] for l in example["label"]]
                label_hot = np.zeros(len(label_list))
                label_hot[label_indices] = 1
                labels.append(label_hot)
        elif output_mode == 'seq_regression':
            labels = [example["label"] for example in examples]
        else:
            labels = [label_to_id[example["label"]] for example in examples]

        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            if output_mode == 'seq_generation':
                feature = InputFeatures(**inputs)
            else:
                feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        # for i, example in enumerate(examples[:3]):
        #     logger.info("*** Example ***")
        #     logger.info(f"id: {example['id']}")
        #     logger.info(f"features: {features[i]}")


    return features
