import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Iterable, Union, Dict

ALL_TORCH_TENSOR_TYPES = (
    torch.Tensor,
    torch.DoubleTensor,
    torch.FloatTensor,
    torch.BFloat16Tensor,
    torch.LongTensor,
    torch.IntTensor,
    torch.ShortTensor,
    torch.HalfTensor,
    torch.CharTensor,
    torch.ByteTensor,
    torch.BoolTensor,
)


class tensor_dict_stack_padding_collater:  # 拼接dict中对应位置的tensor，并padding到最大长度，返回dict
    def __init__(
            self,
            padding_id: Union[int, Dict],
            padding_position: str = "right",
            return_padding_mask: bool = True,
            tensor_keys_to_create_mask: Iterable = None,
            exclude_none: bool = False,
    ):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

        # set keys of tensors in dict to create "padding_mask"
        # if set to "None", then "padding_mask" will be created for all keys in dict
        self.tensor_keys_to_create_mask = set(tensor_keys_to_create_mask)

        # whether to drop the None values in dicts
        self.exclude_none = exclude_none

    def __call__(self, examples):
        """
        examples: list of tensor (or other types) dicts.
        input:
        [
            {
                "key0": int,
                "key1": tensor1,
                "key2": tensor2,
                ...
                "keyN": tensorN,
            },
            {
                "key0": int,
                "key1": tensor1,
                "key2": tensor2,
                ...
                "keyN": tensorN,
            },
            ...
            {
                "key0": int,
                "key1": tensor1,
                "key2": tensor2,
                ...
                "keyN": tensorN,
            }
        ]
        output:
        {
            "key0": [int, int, ..., int],  # Non-tensor values will be returned as a list
            "key1": padded_tensor1,
            "key2": padded_tensor2,
            ...
            "keyN": padded_tensorN,
        }
        """
        keys = examples[0].keys()
        padded_tensors = {}
        padding_masks = {}

        for key in keys:
            value_type = next((type(tensor_dict[key]) for tensor_dict in examples if tensor_dict[key] is not None), type(None))
            padding_value = self.padding_id[key] if isinstance(self.padding_id, dict) else self.padding_id

            if issubclass(value_type, type(None)) and self.exclude_none:  # no valid data, stop here
                return None

            elif issubclass(value_type, ALL_TORCH_TENSOR_TYPES):  # is tensor type, pad as needed
                if self.padding_position == "right":
                    tensors = [tensor_dict[key] for tensor_dict in examples if (self.exclude_none and tensor_dict[key] is not None)]
                    padded_tensor = rnn_utils.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
                elif self.padding_position == "left":  # This will take about twice the time compared to right padding
                    flipped_tensors = [torch.flip(tensor_dict[key], dims=[0]) for tensor_dict in examples if (self.exclude_none and tensor_dict[key] is not None)]
                    flipped_padded_tensors = rnn_utils.pad_sequence(flipped_tensors, batch_first=True, padding_value=padding_value)
                    padded_tensor = torch.flip(flipped_padded_tensors, dims=[1])
                else:
                    raise NotImplementedError

                if self.return_padding_mask:
                    if self.tensor_keys_to_create_mask is None or key in self.tensor_keys_to_create_mask:
                        padding_masks[key] = (padded_tensor != padding_value)
                    else:
                        padding_masks[key] = None

            else:  # not tensor type, return as a list
                padded_tensor = [tensor_dict[key] for tensor_dict in examples if (self.exclude_none and tensor_dict[key] is not None)]

                if self.return_padding_mask:
                    padding_masks[key] = None

            padded_tensors[key] = padded_tensor

        if self.return_padding_mask:
            return padded_tensors, padding_masks
        else:
            return padded_tensors
