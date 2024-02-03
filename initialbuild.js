# trt_llama_api.py
# SPDX-FileCopyrightText: Copyright (c) 2023 
NVIDIA CORPORATION  & 
# AFFILIATES. All
rights reserved.
# SPDX-License-Identifier: MIT
# 
# Permission is hereby granted, 
free of charge, to any person obtaining a # copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# # The above
 copyright notice and this permission notice shall be included in
# all copies or substantial 
portions of the Software. #
# THE SOFTWARE IS PROVIDED
 "AS IS", WITHOUT WARRANTY OF
 ANY KIND, EXPRESS OR # 
IMPLIED, INCLUDING BUT 
NOT LIMITED TO THE
 WARRANTIES OF 
MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER # LIABILITY,
 WHETHER IN AN ACTION
 OF CONTRACT, TORT
 OR OTHERWISE, ARISING
# FROM, OUT 
OF OR IN CONNECTION 
WITH THE 
SOFTWARE 
OR THE USE OR OTHER
# DEALINGS IN 
THE SOFTWARE
--------------------------
import os
from typing 
import 
Any, 
Callable, Dict, \
Optional, Sequence from llama_index.bridge.
pydantic import Field, Private
Attr from llama_index.call
backs import Callback
Manager from llama_index
.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS from llama_index
.llms.base
 import (    ChatMessage,    ChatResponse,    CompletionResponse,    LLMMetadata,   llm_chat_callback,    llm_completion_callback,)
from llama_index.llms
.custom import
 CustomLLM from llama_index.llms
.generic_utils import completion_response_to_chat_response
from llama
_index.llms.generic_utils
 import (  messages_to_prompt
 as generic_messages_to_prompt,)
from
 transformers import
 LlamaTokenizer import gc import json import torch import numpy as np
from tensorrt_llm.runtime importModelConfig, Sampling.Config import tensorrt_llm from pathlib import Path import uuid import time=
EOS_TOKEN = 2  PAD_TOKEN = 2
class TrtLlmAP
I(CustomLLM):
 model_path: Optional

            [str] = Field(  description="The path to the trt engine."
    )
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field
(   description="The maximum
 number of context tokens for
the model."    )
    messages_to_prompt
 Callable = Field
( description="The 
function to convert
 messages to a prompt.", exclude=True )
    completion_to_prompt
Callable = Field
(  description="The function to convert a completion to a prompt.", exclude=True
    )  generate_kwargs: Dict[str, Any] = 
Field(   default_factory=dict, description="Kwargs used for generation." ) model_kwargs: Dict[str, Any] = Field
(  default_factory=dict, description="Kwargs used for model initialization."  )    verbose: bool = Field(description="Whether to print verbose output.")   _model: Any = PrivateAttr()    _model_config: Any = PrivateAttr()    _tokenizer: Any = PrivateAttr()    _max_new_tokens = PrivateAttr()    _sampling_config = PrivateAttr()_verbose = PrivateAttr()   def __init__   self,           model_path: Optional[str] = None,         engine_name: Optional[str] = None,       tokenizer_dir: Optional[str] = None          temperature: float = 0.1        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,    context_window: int = DEFAULT_CONTEXT_WINDOW,
   messages_to_prompt: Optional[Callable] = None,     completion_to_prompt: Optional[Callable] = None,        callback_manager: Optional[CallbackManager] = None,      generate_kwargs: Optional[Dict[str, Any]] = None,         model_kwargs: Optional[Dict[str, Any]] = None,       verbose: bool = False ) -> None:       model_kwargs = model_kwargs or {}      model_kwargs.update({"n_ctx": context_window, "verbose": verbose})      self._max_new_tokens = max_new_tokens    self._verbose = verbose   # check if model is cached       if model_path is not None:      if not os.path.exists(model_path):       raise ValueError(   "Provided model path does not exist. "     "Please check the path or provide a model_url to download."    )        else:      engine_dir = model_path            engine_dir_path = Path(engine_dir)         config_path = engine_dir_path / 'config.json'    
       # config function                with open(config_path, 'r') as f:             config = json.load(f)             use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']            remove_input_padding = config['plugin_config']['remove_input_padding']                tp_size = config['builder_config']['tensor_parallel']            pp_size = config['builder_config']['pipeline_parallel']            world_size = tp_size * pp_size           assert world_size == tensorrt_llm.mpi_world_size(), \                 f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'        num_heads = config['builder_config']['num_heads'] // tp_size           hidden_size = config['builder_config']['hidden_size'] // tp_size     vocab_size = config['builder_config']['vocab_size']      num_layers = config['builder_config']['num_layers']         num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)        paged_kv_cache = config['plugin_config']['paged_kv_cache'           if config['builder_config'].get('multi_query_mode', False):                   tensorrt_llm.logger.warning(       "`multi_query_mode` config is 
deprecated.
 Please
 rebuild: engine."  )                  num_kv_heads = 1            num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
          self._model_config = ModelConfig(num_heads=num_heads,                                                num_kv_heads=num_kv_heads,                                              hidden_size=hidden_size,                                                vocab_size=vocab_size                                        num_layers=num_layers,                                                gpt_attention_plugin=use_gpt_attention_plugin,                                                 paged_kv_cache=paged_kv_cache,                                             remove_input_padding=remove_input_padding               assert pp_size == 1, 'Python runtime does not support pipeline parallelism'               world_size = tp_size * pp_size              runtime_rank = tensorrt_llm.mpi_rank()              runtime_mapping = tensorrt_llm.Mapping(world_size,                                                    runtime_rank,                                           tp_size=tp_size,
                                                   pp_size=pp_size)=    torch.cuda.set_device(runtime_rank%runtime_mapping.gpus_per_node)           self._tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False            self._sampling_config = SamplingConfig(end_id=EOS_TOKEN,                                                       pad_id=PAD_TOKEN,
                                                       num_beams=1,                                                temperature=temperature)
                serialize_path = engine_dir_path / engine_name               with open(serialize_path, 'rb') as f:                  engine_buffer = f.read()                decoder = tensorrt_llm.runtime.GenerationSession(self._model_config,                                                             engine_buffer,                                                              runtime_mapping                                                                debug_mode=False)
                self._model = decoder
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)
        generate_kwargs = generate_kwargs or {}        generate_kwargs.update(            {"temperature": temperature, "max_tokens": max_new_tokens}  )        super().__init__(
            model_path=model_path,
           temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TrtLlmAPI"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self.generate_kwargs.update({"stream": False})

        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        input_text = prompt
        input_ids, input_lengths = self.parse_input(input_text, self._tokenizer,
                                                    EOS_TOKEN,
                                                    self._model_config)

        max_input_length = torch.max(input_lengths).item()
        self._model.setup(input_lengths.size(0), max_input_length, self._max_new_tokens, 1) # beam size is set to 1
        if self._verbose:
            start_time = time.time()

        output_ids = self._model.decode(input_ids, input_lengths, self._sampling_config)
        torch.cuda.synchronize()

        elapsed_time = None
        if self._verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time


        output_txt, output_token_ids = self.get_output(output_ids,
                                       input_lengths,
                                       self._max_new_tokens,
                                       self._tokenizer)

        if self._verbose:
            print(f"Input context length  : {input_ids.shape[1]}")
            print(f"Inference time        : {elapsed_time:.2f} seconds")
            print(f"Output context length : {len(output_token_ids)} ")
            print(f"Inference token/sec   : {(len(output_token_ids) / elapsed_time):2f}")

        # call garbage collected after inference
        torch.cuda.empty_cache()
        gc.collect()

        return CompletionResponse(text=output_txt, raw=self.generate_completion_dict(output_txt))

    def parse_input(self, input_text: str, tokenizer, end_id: int,
                    remove_input_padding: bool):
        input_tokens = []

        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))

        input_lengths = torch.tensor([len(x) for x in input_tokens],
                                     dtype=torch.int32,
                                     device='cuda')
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                     device='cuda').unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                end_id).cuda()

        return input_ids, input_lengths

    def remove_extra_eos_ids(self, outputs):
        outputs.reverse()
        while outputs and outputs[0] == 2:
            outputs.pop(0)
        outputs.reverse()
        outputs.append(2)
        return outputs

    def get_output(self, output_ids, input_lengths, max_output_len, tokenizer):
        num_beams = output_ids.size(1)
        output_text = ""
        outputs = None
        for b in range(input_lengths.size(0)):
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                outputs = self.remove_extra_eos_ids(outputs)
                output_text = tokenizer.decode(outputs)

        return output_text, outputs

    def generate_completion_dict(self, text_str):
        """
        Generate a dictionary for text completion details.
        Returns:
        dict: A dictionary containing completion details.
        """
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        model_name: str = self._model if self._model is not None else self.model_path
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": 'stop'
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        pass

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import ctypes
import platform
from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import tensorrt as trt

from tensorrt_llm.logger import logger

from .._ipc_utils import IpcMemory
from ..mapping import Mapping

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'


def plugin_lib_path() -> str:
    project_dir = Path(__file__).parent.parent.absolute()
    dyn_lib = "libnvinfer_plugin_tensorrt_llm.so" if platform.system(
    ) != "Windows" else "nvinfer_plugin_tensorrt_llm.dll"
    return str(project_dir.joinpath("libs", dyn_lib))


def _load_plugin_lib():
    winmode = 0 if platform.system() == "Windows" else None
    handle = ctypes.CDLL(plugin_lib_path(),
                         mode=ctypes.RTLD_GLOBAL,
                         winmode=winmode)
    try:
        handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtLlmPlugins.restype = ctypes.c_bool
    except AttributeError as err:
        raise ImportError('TensorRT-LLM Plugin is unavailable') from err
    assert handle.initTrtLlmPlugins(None,
                                    TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))


class ContextFMHAType(IntEnum):
    disabled = 0
    # FP16 I/O, FP16 Accumulation
    enabled = 1
    # FP16 I/O, FP32 Accumulation
    enabled_with_fp32_acc = 2


@dataclass
class PluginConfig:

    # Plugins
    bert_attention_plugin: str = "float16"
    gpt_attention_plugin: str = "float16"
    gemm_plugin: str = None
    smooth_quant_gemm_plugin: str = None
    identity_plugin: str = None
    layernorm_quantization_plugin: str = None
    rmsnorm_quantization_plugin: str = None
    nccl_plugin: str = "float16"
    lookup_plugin: str = None
    lora_plugin: str = None
    weight_only_groupwise_quant_matmul_plugin: str = None
    weight_only_quant_matmul_plugin: str = None
    quantize_per_token_plugin: bool = False
    quantize_tensor_plugin: bool = False

    # Features
    context_fmha: bool = True
    context_fmha_fp32_acc: bool = False  # will use fp16 if disabled
    paged_kv_cache: bool = True
    remove_input_padding: bool = True
    # TODO[kevin]: smart strategy to choose all reduce plugin
    use_custom_all_reduce: bool = True
    multi_block_mode: bool = False
    enable_xqa: bool = True
    attention_qk_half_accumulation: bool = False
    tokens_per_block: int = 128
    use_paged_context_fmha: bool = False
    use_context_fmha_for_generation: bool = False

    def set_plugin(self, name: str, value: Union[str, bool, int]):
        assert hasattr(self, name), f"Plugin name doesn't exist: {name}"
        if value is not None and getattr(self, name) is not None:
            target_type = type(getattr(self, name))
            assert type(value) == target_type, \
                f"Plugin {name} expects {target_type}, got {type(value)}"
        setattr(self, name, value)
        logger.info(f"Set {name} to {value}.")

    def update_from_dict(self, config: dict):
        for name in config.keys():
            if hasattr(self, name):
                value_to_be_update = config[name]
                if type(getattr(self, name)) == bool:
                    if value_to_be_update is True or \
                            value_to_be_update == "enable":
                        value_to_be_update = True
                    elif value_to_be_update is False or \
                            value_to_be_update == "disable":
                        value_to_be_update = False
                    else:
                        raise RuntimeError(
                            f"Unexpected value ({value_to_be_update}) to be updated for {name}."
                        )
                elif value_to_be_update == "disable":
                    value_to_be_update = None
                self.set_plugin(name, value_to_be_update)

    @classmethod
    def from_dict(cls, config: dict):
        plugin_config = cls()
        plugin_config.update_from_dict(config)
        return plugin_config

    @classmethod
    def from_arguments(cls, args: argparse.Namespace):
        return cls.from_dict(vars(args))

    def to_legacy_setting(self):
        '''Legacy setting means that all of the plugins and features are
        disabled, this needed for the legacy `build.py` script, which will be
        migrated to the centralized building script `tensorrt_llm/commands/build.py`.

        After the migration is done, this function may or may not be deleted.
        '''
        for field in fields(self):
            if field.type == str:
                self.set_plugin(field.name, None)
            elif field.type == bool:
                self.set_plugin(field.name, False)

    @property
    def context_fmha_type(self):
        if self.context_fmha:
            if self.context_fmha_fp32_acc:
                return ContextFMHAType.enabled_with_fp32_acc
            else:
                return ContextFMHAType.enabled
        else:
            return ContextFMHAType.disabled

    @context_fmha_type.setter
    def context_fmha_type(self, value):
        if value == ContextFMHAType.disabled:
            self.set_plugin("context_fmha", False)
        else:
            self.set_plugin("context_fmha", True)
            if value == ContextFMHAType.enabled:
                self.set_plugin("context_fmha_fp32_acc", False)
            elif value == ContextFMHAType.enabled_with_fp32_acc:
                self.set_plugin("context_fmha_fp32_acc", True)

    def set_smooth_quant_plugins(self, dtype: str = "float16"):
        self.set_plugin("smooth_quant_gemm_plugin", dtype)
        self.set_plugin("rmsnorm_quantization_plugin", dtype)
        self.set_plugin("layernorm_quantization_plugin", dtype)
        self.set_plugin("quantize_per_token_plugin", True)
        self.set_plugin("quantize_tensor_plugin", True)

    def enable_qk_half_accum(self):
        self.set_plugin("attention_qk_half_accumulation", True)
        return self

    def set_context_fmha(self, context_fmha_type=ContextFMHAType.enabled):
        assert type(context_fmha_type) == ContextFMHAType
        self.context_fmha_type = context_fmha_type
        return self

    def enable_remove_input_padding(self):
        self.set_plugin("remove_input_padding", True)
        return self

    def enable_paged_kv_cache(self, tokens_per_block=128):
        self.set_plugin("paged_kv_cache", True)
        self.set_plugin("tokens_per_block", tokens_per_block)
        return self

    def set_gpt_attention_plugin(self, dtype='float16':
        self.set_plugin"gpt_attention_plugin", dtype
        return self
   def enable_mmha_multi_block_mode(self):
        self.set_plugin"multi_block_mode", True
        return self
    def enable_xqa_optimization(self):
        self.set_plugi"enable_xqa", True
        return self
    def set_bert_attention_plugin(self, dtype='float16'):
        self.set_plugin "bert_attention_plugin ", dtype

        return self
    def set_identity_plugin(self, dtype= 'float16:
        self.set_plugin "identity_plugin", dtype
       return self
    def set_gemm_pluginself, 
dtype='float16':     self.set_plugin"gemm_plugin", dtype
        return self
    def set_smooth_quant_gemm_plugin(self, dtype='float16'):
        self.set_plugin("smooth_quant_gemm_plugin", dtype)
        return self
    def set_layernorm_quantization_plugin(self, dtype='float16':
        self.set_plugin
"layernorm_quantization_plugin", dtype
        return self
    def set_rmsnorm_quantization_plugin(self, dtype='float16'):
        self.set_plugin
"rmsnorm_quantization_plugin", dtype
        return self
    def 
set_weight_only_quant_matmul_plugin(self, dtype='float16':       self.set_plugin("weight_only_quant_matmul_plugin", dtype
        return self
   def set_weight_only_groupwise_quant_matmul_plugin(self, dtype='float16'):self.set_plugin("weight_only_groupwise_quant_matmul_plugin", dtype)
        return self
    def set_nccl_plugin(self,
                        dtype='float16',               
   use_custom_all_reduce: bool = False):
        self.set_plugin("nccl_plugin", dtype)
        self.set_plugin
"use_custom_all_reduce", use_custom_all_reduce
        if use_custom_all_reduce:
            init_all_reduce_helper()
        return self
    def set_quantize_per_token_plugin(self):
        self.set_plugin
"quantize_per_token_plugin", True
        return self
    def set_quantize_tensor_plugin(self):       
 self.set_plugin("quantize_tensor_plugin", True)
        return self
    def set_lookup_plugin(self, dtype='float16'):
        self.set_plugin("lookup_plugin", dtype)
        return self
    def set_lora_plugin(self, dtype='float16'):
        self.set_plugin("lora_plugin", dtype)
        return self
    def set_paged_context_fmha(self):
        self.set_plugin
"use_paged_context_fmha", True        return self 
    def set_context_fmha_for_generation(self):
        self.set_plugin
"use_context_fmha_for_generation", True ,  return self
cli_plugin_args =   [   # Plugins  "bert_attention_plugin","gpt_attention_plugin", "gemm_plugin","lookup_plugin",    "lora_plugin",

  # Features
   "context_fmha","context_fmha_fp32_acc "paged_kv_cache","remove_input_padding",  "use_custom_all_reduce",  "multi_block_mode",   "enable_xqa",   "attention_qk_half_accumulation",  "tokens_per_block",  "use_paged_context_fmha",   "use_context_fmha_for_generation" ], plugin_options =['float_16' ] , ['float_32'] , 
['bfloat_16'] , ['disable']
def add_plugin_argument(parser):    plugin_config = PluginConfig()    for field in fields (plugin_config):if field.name not
 in cli_plugin_args:            continue      if field.type ==
 str: parser.add_argument
(
                (" - - ") +  field.name , type=str , default=field.default if field.default is not None,
 else No choices=plugin_options) elif field.type ==
 bool: parser.add_argument    ( 
 (" - - ")  + field.name, type=str,
    default=['enable'] if field.default else ['disable']choices=[' enable' ],
[ 'disable ']
) ,  else parser.add_argument( " - - " ) 
+ field.name,  type=field.type,          default=field.default)
    return parser  class CustomAllReduceHelper: (" " ") Globally visible class to help usage of custom_all_reduce plugin Provides the following utilities:gen_id_int:  Used for synchronization with custom kernels. Plugins instances MUST have the same
  id across GPUs.
{example : GPU[#zeros] all  reduce after MLP at layer i must have the same id as
 GPU [#1], GPU [#2] , Also, ids MUST be unique per model. There should not be two all 
reduce instances in GPU [#0]that have the 
same id.   
workspace: Tensor When using [CUSTOM] or [AUTO] mode, a tensor containing pointers to memory   visible to all GPUs. It should be ( 3 ) poitners per TP rank -   ptr to data buffer, ptr to barriers in, ptr to barriers out.   It must be initialized using IpcMemory class.  Usage:
 Use `init_all_reduce_helper` to reset the id counter. 
This must be done in main model class_Set.custom_all_reduce_helper.workspace with the required tensor.  Then, each instance of allreduce will reference that tensor automatically.  { " " " } POINTERS_PER_ RANK = ( 4 ) def  _ _ init _ _ (self)  -> None:   self.current_id: {int = 1} self.workspace: Optional[Tensor] = None   def gen_id(self) -> int:  result = self.current_id   self.current_id {+ = 1 }   return result   def set_workspace_tensor(self,   mapping: Mapping,   two_opt_profiles: Optional[bool] = None): from ..functional import Tensor    workspace_size=self.POINTERS_PER_RANK * mapping.tp_size  dim_range = Non  if two_opt_profiles is not None:dim_range = OrderedDict([ ('all_reduce_size', [workspace_size, workspace_size]
 if two_opt_profiles else [workspace_size])  ])  self.workspace = Tensor(   name='all_reduce_workspace', dtype=trt.int64,   shape=[workspace_size]  dim_range=dim_range,  )  @staticmethod  def max_workspace_size_auto(tp_size: int) -> int: if tp_size                                            <= 2:   return
 16_000_000  return
   8_000_000
  @static method 
  def allocate_workspace(mapping: Mapping,
  size: int)                     -> Tuple[List[IpcMemory], "torch.tensor"]:import torch        ipc_buffers_ping=IpcMemory(mapping, size)ipc_buffers_pong=IpcMemory(mapping, size)
  ipc_barriers_in = IpcMemory(  mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size) ipc_barriers_out = IpcMemory ( mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size) , buffers = [ ipc_buffers_ping,
                 ipc_buffers_pong, ipc_barriers_in,
            ipc_buffers_ping] return 
buffers, torch.tensor
( ipc_buffers_ping.serialize() 
        + ipc_buffers_pong.serialize()
 + ipc_barriers_in.serialize() 
        + ipc_barriers_out.serialize(),          dtype=torch.int64,
  device="cpu")
custom_all_reduce_helper = None
def init_all_reduce_helper():
    global custom_all_reduce_helper
    custom_all_reduce_helper = CustomAllReduceHelper()
def current_all_reduce_helper():
    global custom_all_reduce_helper
    assert custom_all_reduce_helper is not None, "You must call `init_all_reduce_helper` first"
    return custom_all_reduce_helper
trt_llama_api.py # SPDX-FileCopyrightText: 
Copyright (c) 2023 
NVIDIA CORPORATION  & #
 AFFILIATES. All rights reserved. #
 SPDX-License-Identifier: 
MIT # 
# Permission is hereby granted, 
free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# # The above
 copyright notice and this permission notice shall be included in
# all copies or substantial 
portions of the Software. #
# THE SOFTWARE IS PROVIDED
 "AS IS", WITHOUT WARRANTY OF
 ANY KIND, EXPRESS OR # 
IMPLIED, INCLUDING BUT 
NOT LIMITED TO THE
 WARRANTIES OF 
MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER # LIABILITY,
 WHETHER IN AN ACTION
 OF CONTRACT, TORT
 OR OTHERWISE, ARISING
# FROM, OUT 
OF OR IN CONNECTION 
WITH THE 
SOFTWARE 
OR THE USE OR OTHER
# DEALINGS IN 
THE SOFTWARE
--------------------------
import os
from typing 
import 
Any, 
Callable, Dict, \
Optional, Sequence from llama_index.bridge.
pydantic import Field, Private
Attr from llama_index.call
backs import Callback
Manager from llama_index
.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS from llama_index
.llms.base
 import (    ChatMessage,    ChatResponse,    CompletionResponse,    LLMMetadata,   llm_chat_callback,    llm_completion_callback,)
from llama_index.llms
.custom import
 CustomLLM from llama_index.llms
.generic_utils import completion_response_to_chat_response
from llama
_index.llms.generic_utils
 import (  messages_to_prompt
 as generic_messages_to_prompt,)
from
 transformers import
 LlamaTokenizer import gc import json import torch import numpy as np
from tensorrt_llm.runtime importModelConfig, Sampling.Config import tensorrt_llm from pathlib import Path import uuid import time=
EOS_TOKEN = 2  PAD_TOKEN = 2
class TrtLlmAP
I(CustomLLM):
 model_path: Optional

            [str] = Field(  description="The path to the trt engine."
    )
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field
(   description="The maximum
 number of context tokens for
the model."    )
    messages_to_prompt
 Callable = Field
( description="The 
function to convert
 messages to a prompt.", exclude=True )
    completion_to_prompt
Callable = Field
(  description="The function to convert a completion to a prompt.", exclude=True
    )  generate_kwargs: Dict[str, Any] = 
Field(   default_factory=dict, description="Kwargs used for generation." ) model_kwargs: Dict[str, Any] = Field
(  default_factory=dict, description="Kwargs used for model initialization."  )    verbose: bool = Field(description="Whether to print verbose output.")   _model: Any = PrivateAttr()    _model_config: Any = PrivateAttr()    _tokenizer: Any = PrivateAttr()    _max_new_tokens = PrivateAttr()    _sampling_config = PrivateAttr()_verbose = PrivateAttr()   def __init__   self,           model_path: Optional[str] = None,         engine_name: Optional[str] = None,       tokenizer_dir: Optional[str] = None          temperature: float = 0.1        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,    context_window: int = DEFAULT_CONTEXT_WINDOW,
   messages_to_prompt: Optional[Callable] = None,     completion_to_prompt: Optional[Callable] = None,        callback_manager: Optional[CallbackManager] = None,      generate_kwargs: Optional[Dict[str, Any]] = None,         model_kwargs: Optional[Dict[str, Any]] = None,       verbose: bool = False ) -> None:       model_kwargs = model_kwargs or {}      model_kwargs.update({"n_ctx": context_window, "verbose": verbose})      self._max_new_tokens = max_new_tokens    self._verbose = verbose   # check if model is cached       if model_path is not None:      if not os.path.exists(model_path):       raise ValueError(   "Provided model path does not exist. "     "Please check the path or provide a model_url to download."    )        else:      engine_dir = model_path            engine_dir_path = Path(engine_dir)         config_path = engine_dir_path / 'config.json'    
       # config function                with open(config_path, 'r') as f:             config = json.load(f)             use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']            remove_input_padding = config['plugin_config']['remove_input_padding']                tp_size = config['builder_config']['tensor_parallel']            pp_size = config['builder_config']['pipeline_parallel']            world_size = tp_size * pp_size           assert world_size == tensorrt_llm.mpi_world_size(), \                 f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'        num_heads = config['builder_config']['num_heads'] // tp_size           hidden_size = config['builder_config']['hidden_size'] // tp_size     vocab_size = config['builder_config']['vocab_size']      num_layers = config['builder_config']['num_layers']         num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)        paged_kv_cache = config['plugin_config']['paged_kv_cache'           if config['builder_config'].get('multi_query_mode', False):                   tensorrt_llm.logger.warning(       "`multi_query_mode` config is 
deprecated.
 Please
 rebuild: engine."  )                  num_kv_heads = 1            num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
          self._model_config = ModelConfig(num_heads=num_heads,                                                num_kv_heads=num_kv_heads,                                              hidden_size=hidden_size,                                                vocab_size=vocab_size                                        num_layers=num_layers,                                                gpt_attention_plugin=use_gpt_attention_plugin,                                                 paged_kv_cache=paged_kv_cache,                                             remove_input_padding=remove_input_padding               assert pp_size == 1, 'Python runtime does not support pipeline parallelism'               world_size = tp_size * pp_size              runtime_rank = tensorrt_llm.mpi_rank()              runtime_mapping = tensorrt_llm.Mapping(world_size,                                                    runtime_rank,                                           tp_size=tp_size,
                                                   pp_size=pp_size)=    torch.cuda.set_device(runtime_rank%runtime_mapping.gpus_per_node)           self._tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False            self._sampling_config = SamplingConfig(end_id=EOS_TOKEN,                                                       pad_id=PAD_TOKEN,
                                                       num_beams=1,                                                temperature=temperature)
                serialize_path = engine_dir_path / engine_name               with open(serialize_path, 'rb') as f:                  engine_buffer = f.read()                decoder = tensorrt_llm.runtime.GenerationSession(self._model_config,                                                             engine_buffer,                                                              runtime_mapping                                                                debug_mode=False)
                self._model = decoder
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)
        generate_kwargs = generate_kwargs or {}        generate_kwargs.update(            {"temperature": temperature, "max_tokens": max_new_tokens}  )        super().__init__(
            model_path=model_path,
           temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TrtLlmAPI"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self.generate_kwargs.update({"stream": False})

        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        input_text = prompt
        input_ids, input_lengths = self.parse_input(input_text, self._tokenizer,
                                                    EOS_TOKEN,
                                                    self._model_config)

        max_input_length = torch.max(input_lengths).item()
        self._model.setup(input_lengths.size(0), max_input_length, self._max_new_tokens, 1) # beam size is set to 1
        if self._verbose:
            start_time = time.time()

        output_ids = self._model.decode(input_ids, input_lengths, self._sampling_config)
        torch.cuda.synchronize()

        elapsed_time = None
        if self._verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time


        output_txt, output_token_ids = self.get_output(output_ids,
                                       input_lengths,
                                       self._max_new_tokens,
                                       self._tokenizer)

        if self._verbose:
            print(f"Input context length  : {input_ids.shape[1]}")
            print(f"Inference time        : {elapsed_time:.2f} seconds")
            print(f"Output context length : {len(output_token_ids)} ")
            print(f"Inference token/sec   : {(len(output_token_ids) / elapsed_time):2f}")

        # call garbage collected after inference
        torch.cuda.empty_cache()
        gc.collect()

        return CompletionResponse(text=output_txt, raw=self.generate_completion_dict(output_txt))

    def parse_input(self, input_text: str, tokenizer, end_id: int,
                    remove_input_padding: bool):
        input_tokens = []

        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))

        input_lengths = torch.tensor([len(x) for x in input_tokens],
                                     dtype=torch.int32,
                                     device='cuda')
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                     device='cuda').unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                end_id).cuda()

        return input_ids, input_lengths

    def remove_extra_eos_ids(self, outputs):
        outputs.reverse()
        while outputs and outputs[0] == 2:
            outputs.pop(0)
        outputs.reverse()
        outputs.append(2)
        return outputs

    def get_output(self, output_ids, input_lengths, max_output_len, tokenizer):
        num_beams = output_ids.size(1)
        output_text = ""
        outputs = None
        for b in range(input_lengths.size(0)):
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                outputs = self.remove_extra_eos_ids(outputs)
                output_text = tokenizer.decode(outputs)

        return output_text, outputs

    def generate_completion_dict(self, text_str):
        """
        Generate a dictionary for text completion details.
        Returns:
        dict: A dictionary containing completion details.
        """
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        model_name: str = self._model if self._model is not None else self.model_path
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": 'stop'
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        pass

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import ctypes
import platform
from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import tensorrt as trt

from tensorrt_llm.logger import logger

from .._ipc_utils import IpcMemory
from ..mapping import Mapping

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'


def plugin_lib_path() -> str:
    project_dir = Path(__file__).parent.parent.absolute()
    dyn_lib = "libnvinfer_plugin_tensorrt_llm.so" if platform.system(
    ) != "Windows" else "nvinfer_plugin_tensorrt_llm.dll"
    return str(project_dir.joinpath("libs", dyn_lib))


def _load_plugin_lib():
    winmode = 0 if platform.system() == "Windows" else None
    handle = ctypes.CDLL(plugin_lib_path(),
                         mode=ctypes.RTLD_GLOBAL,
                         winmode=winmode)
    try:
        handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtLlmPlugins.restype = ctypes.c_bool
    except AttributeError as err:
        raise ImportError('TensorRT-LLM Plugin is unavailable') from err
    assert handle.initTrtLlmPlugins(None,
                                    TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))


class ContextFMHAType(IntEnum):
    disabled = 0
    # FP16 I/O, FP16 Accumulation
    enabled = 1
    # FP16 I/O, FP32 Accumulation
    enabled_with_fp32_acc = 2


@dataclass
class PluginConfig:

    # Plugins
    bert_attention_plugin: str = "float16"
    gpt_attention_plugin: str = "float16"
    gemm_plugin: str = None
    smooth_quant_gemm_plugin: str = None
    identity_plugin: str = None
    layernorm_quantization_plugin: str = None
    rmsnorm_quantization_plugin: str = None
    nccl_plugin: str = "float16"
    lookup_plugin: str = None
    lora_plugin: str = None
    weight_only_groupwise_quant_matmul_plugin: str = None
    weight_only_quant_matmul_plugin: str = None
    quantize_per_token_plugin: bool = False
    quantize_tensor_plugin: bool = False

    # Features
    context_fmha: bool = True
    context_fmha_fp32_acc: bool = False  # will use fp16 if disabled
    paged_kv_cache: bool = True
    remove_input_padding: bool = True
    # TODO[kevin]: smart strategy to choose all reduce plugin
    use_custom_all_reduce: bool = True
    multi_block_mode: bool = False
    enable_xqa: bool = True
    attention_qk_half_accumulation: bool = False
    tokens_per_block: int = 128
    use_paged_context_fmha: bool = False
    use_context_fmha_for_generation: bool = False

    def set_plugin(self, name: str, value: Union[str, bool, int]):
        assert hasattr(self, name), f"Plugin name doesn't exist: {name}"
        if value is not None and getattr(self, name) is not None:
            target_type = type(getattr(self, name))
            assert type(value) == target_type, \
                f"Plugin {name} expects {target_type}, got {type(value)}"
        setattr(self, name, value)
        logger.info(f"Set {name} to {value}.")

    def update_from_dict(self, config: dict):
        for name in config.keys():
            if hasattr(self, name):
                value_to_be_update = config[name]
                if type(getattr(self, name)) == bool:
                    if value_to_be_update is True or \
                            value_to_be_update == "enable":
                        value_to_be_update = True
                    elif value_to_be_update is False or \
                            value_to_be_update == "disable":
                        value_to_be_update = False
                    else:
                        raise RuntimeError(
                            f"Unexpected value ({value_to_be_update}) to be updated for {name}."
                        )
                elif value_to_be_update == "disable":
                    value_to_be_update = None
                self.set_plugin(name, value_to_be_update)

    @classmethod
    def from_dict(cls, config: dict):
        plugin_config = cls()
        plugin_config.update_from_dict(config)
        return plugin_config

    @classmethod
    def from_arguments(cls, args: argparse.Namespace):
        return cls.from_dict(vars(args))

    def to_legacy_setting(self):
        '''Legacy setting means that all of the plugins and features are
        disabled, this needed for the legacy `build.py` script, which will be
        migrated to the centralized building script `tensorrt_llm/commands/build.py`.

        After the migration is done, this function may or may not be deleted.
        '''
        for field in fields(self):
            if field.type == str:
                self.set_plugin(field.name, None)
            elif field.type == bool:
                self.set_plugin(field.name, False)

    @property
    def context_fmha_type(self):
        if self.context_fmha:
            if self.context_fmha_fp32_acc:
                return ContextFMHAType.enabled_with_fp32_acc
            else:
                return ContextFMHAType.enabled
        else:
            return ContextFMHAType.disabled

    @context_fmha_type.setter
    def context_fmha_type(self, value):
        if value == ContextFMHAType.disabled:
            self.set_plugin("context_fmha", False)
        else:
            self.set_plugin("context_fmha", True)
            if value == ContextFMHAType.enabled:
                self.set_plugin("context_fmha_fp32_acc", False)
            elif value == ContextFMHAType.enabled_with_fp32_acc:
                self.set_plugin("context_fmha_fp32_acc", True)

    def set_smooth_quant_plugins(self, dtype: str = "float16"):
        self.set_plugin("smooth_quant_gemm_plugin", dtype)
        self.set_plugin("rmsnorm_quantization_plugin", dtype)
        self.set_plugin("layernorm_quantization_plugin", dtype)
        self.set_plugin("quantize_per_token_plugin", True)
        self.set_plugin("quantize_tensor_plugin", True)

    def enable_qk_half_accum(self):
        self.set_plugin("attention_qk_half_accumulation", True)
        return self

    def set_context_fmha(self, context_fmha_type=ContextFMHAType.enabled):
        assert type(context_fmha_type) == ContextFMHAType
        self.context_fmha_type = context_fmha_type
        return self

    def enable_remove_input_padding(self):
        self.set_plugin("remove_input_padding", True)
        return self

    def enable_paged_kv_cache(self, tokens_per_block=128):
        self.set_plugin("paged_kv_cache", True)
        self.set_plugin("tokens_per_block", tokens_per_block)
        return self

    def set_gpt_attention_plugin(self, dtype='float16':
        self.set_plugin"gpt_attention_plugin", dtype
        return self
   def enable_mmha_multi_block_mode(self):
        self.set_plugin"multi_block_mode", True
        return self
    def enable_xqa_optimization(self):
        self.set_plugi"enable_xqa", True
        return self
    def set_bert_attention_plugin(self, dtype='float16'):
        self.set_plugin "bert_attention_plugin ", dtype

        return self
    def set_identity_plugin(self, dtype= 'float16:
        self.set_plugin "identity_plugin", dtype
       return self
    def set_gemm_pluginself, 
dtype='float16':     self.set_plugin"gemm_plugin", dtype
        return self
    def set_smooth_quant_gemm_plugin(self, dtype='float16'):
        self.set_plugin("smooth_quant_gemm_plugin", dtype)
        return self
    def set_layernorm_quantization_plugin(self, dtype='float16':
        self.set_plugin
"layernorm_quantization_plugin", dtype
        return self
    def set_rmsnorm_quantization_plugin(self, dtype='float16'):
        self.set_plugin
"rmsnorm_quantization_plugin", dtype
        return self
    def 
set_weight_only_quant_matmul_plugin(self, dtype='float16':       self.set_plugin("weight_only_quant_matmul_plugin", dtype
        return self
   def set_weight_only_groupwise_quant_matmul_plugin(self, dtype='float16'):self.set_plugin("weight_only_groupwise_quant_matmul_plugin", dtype)
        return self
    def set_nccl_plugin(self,
                        dtype='float16',               
   use_custom_all_reduce: bool = False):
        self.set_plugin("nccl_plugin", dtype)
        self.set_plugin
"use_custom_all_reduce", use_custom_all_reduce
        if use_custom_all_reduce:
            init_all_reduce_helper()
        return self
    def set_quantize_per_token_plugin(self):
        self.set_plugin
"quantize_per_token_plugin", True
        return self
    def set_quantize_tensor_plugin(self):       
 self.set_plugin("quantize_tensor_plugin", True)
        return self
    def set_lookup_plugin(self, dtype='float16'):
        self.set_plugin("lookup_plugin", dtype)
        return self
    def set_lora_plugin(self, dtype='float16'):
        self.set_plugin("lora_plugin", dtype)
        return self
    def set_paged_context_fmha(self):
        self.set_plugin
"use_paged_context_fmha", True        return self 
    def set_context_fmha_for_generation(self):
        self.set_plugin
"use_context_fmha_for_generation", True ,  return self
cli_plugin_args =   [   # Plugins  "bert_attention_plugin","gpt_attention_plugin", "gemm_plugin","lookup_plugin",    "lora_plugin",

  # Features
   "context_fmha","context_fmha_fp32_acc "paged_kv_cache","remove_input_padding",  "use_custom_all_reduce",  "multi_block_mode",   "enable_xqa",   "attention_qk_half_accumulation",  "tokens_per_block",  "use_paged_context_fmha",   "use_context_fmha_for_generation" ], plugin_options =['float_16' ] , ['float_32'] , 
['bfloat_16'] , ['disable']
def add_plugin_argument(parser):    plugin_config = PluginConfig()    for field in fields (plugin_config):if field.name not
 in cli_plugin_args:            continue      if field.type ==
 str: parser.add_argument
(
                (" - - ") +  field.name , type=str , default=field.default if field.default is not None,
 else No choices=plugin_options) elif field.type ==
 bool: parser.add_argument    ( 
 (" - - ")  + field.name, type=str,
    default=['enable'] if field.default else ['disable']choices=[' enable' ],
[ 'disable ']
) ,  else parser.add_argument( " - - " ) 
+ field.name,  type=field.type,          default=field.default)
    return parser  class CustomAllReduceHelper: (" " ") Globally visible class to help usage of custom_all_reduce plugin Provides the following utilities:gen_id_int:  Used for synchronization with custom kernels. Plugins instances MUST have the same
  id across GPUs.
{example : GPU[#zeros] all  reduce after MLP at layer i must have the same id as
 GPU [#1], GPU [#2] , Also, ids MUST be unique per model. There should not be two all 
reduce instances in GPU [#0]that have the 
same id.   
workspace: Tensor When using [CUSTOM] or [AUTO] mode, a tensor containing pointers to memory   visible to all GPUs. It should be ( 3 ) poitners per TP rank -   ptr to data buffer, ptr to barriers in, ptr to barriers out.   It must be initialized using IpcMemory class.  Usage:
 Use `init_all_reduce_helper` to reset the id counter. 
This must be done in main model class_Set.custom_all_reduce_helper.workspace with the required tensor.  Then, each instance of allreduce will reference that tensor automatically.  { " " " } POINTERS_PER_ RANK = ( 4 ) def  _ _ init _ _ (self)  -> None:   self.current_id: {int = 1} self.workspace: Optional[Tensor] = None   def gen_id(self) -> int:  result = self.current_id   self.current_id {+ = 1 }   return result   def set_workspace_tensor(self,   mapping: Mapping,   two_opt_profiles: Optional[bool] = None): from ..functional import Tensor    workspace_size=self.POINTERS_PER_RANK * mapping.tp_size  dim_range = Non  if two_opt_profiles is not None:dim_range = OrderedDict([ ('all_reduce_size', [workspace_size, workspace_size]
 if two_opt_profiles else [workspace_size])  ])  self.workspace = Tensor(   name='all_reduce_workspace', dtype=trt.int64,   shape=[workspace_size]  dim_range=dim_range,  )  @staticmethod  def max_workspace_size_auto(tp_size: int) -> int: if tp_size                                            <= 2:   return
 16_000_000  return
   8_000_000
  @static method 
  def allocate_workspace(mapping: Mapping,
  size: int)                     -> Tuple[List[IpcMemory], "torch.tensor"]:import torch        ipc_buffers_ping=IpcMemory(mapping, size)ipc_buffers_pong=IpcMemory(mapping, size)
  ipc_barriers_in = IpcMemory(  mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size) ipc_barriers_out = IpcMemory ( mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size) , buffers = [ ipc_buffers_ping,
                 ipc_buffers_pong, ipc_barriers_in,
            ipc_buffers_ping] return 
buffers, torch.tensor
( ipc_buffers_ping.serialize() 
        + ipc_buffers_pong.serialize()
 + ipc_barriers_in.serialize() 
        + ipc_barriers_out.serialize(),          dtype=torch.int64,
  device="cpu")
custom_all_reduce_helper = None
def init_all_reduce_helper():
    global custom_all_reduce_helper
    custom_all_reduce_helper = CustomAllReduceHelper()
def current_all_reduce_helper():
    global custom_all_reduce_helper
    assert custom_all_reduce_helper is not None, "You must call `init_all_reduce_helper` first"
    return custom_all_reduce_helper
