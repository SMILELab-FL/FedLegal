# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FedLab communication API"""
HEADER_SENDER_RANK_IDX = 0
HEADER_RECEIVER_RANK_IDX = 1
HEADER_SLICE_SIZE_IDX = 2
HEADER_MESSAGE_CODE_IDX = 3
HEADER_DATA_TYPE_IDX = 4

DEFAULT_RECEIVER_RANK = -1
DEFAULT_SLICE_SIZE = 0
DEFAULT_MESSAGE_CODE_VALUE = 0

HEADER_SIZE = 5

# DATA TYPE CONSTANT
INT8 = 0
INT16 = 1
INT32 = 2
INT64 = 3

FLOAT16 = 4
FLOAT32 = 5
FLOAT64 = 6


def dtype_torch2flab(torch_type):
    return supported_torch_dtypes.index(torch_type)

def dtype_flab2torch(fedlab_type):
    return supported_torch_dtypes[fedlab_type]

from .package import Package, supported_torch_dtypes
from .processor import PackageProcessor
