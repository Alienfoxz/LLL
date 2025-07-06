"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
from itertools import zip_longest

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    if "qwen" in tokenizer.name_or_path.lower():
        special_token = tokenizer.encode("<|im_start|>")[0]
        reward_token = tokenizer.encode("<|im_end|>")[0]
    elif "llama-3" in tokenizer.name_or_path.lower():
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1] # remove the last token
    response_mask = response_mask[:, :-1] # remove the last token

    return score_tensor, loss_mask, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Args:
            config: Configuration object
            tokenizer: Tokenizer for text processing
            processor: Optional processor for multi-modal data
            mode: "train" or "val" mode
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
        self.mode = mode
        self.es_cfg = self.config.es_manager[mode]
        
        # Initialize environment configurations
        self._init_env_configs()
        
    def _init_env_configs(self):
        """Initialize environment configurations and prefixes"""
        self.env_configs = {}
        self.prefix_lookup = {}
        
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue
                
            # Get base environment configuration
            base_config = REGISTERED_ENV_CONFIGS[env_config.env_type]()
            env_config_new = asdict(base_config)
            
            # Update with custom configurations
            for k, v in env_config.items():
                env_config_new[k] = v
                
            # Build environment instruction
            env_instruction = self._build_env_instruction(env_config_new)
            
            # Store configurations
            print("env_tag", env_tag, env_instruction)
            self.env_configs[env_tag] = env_config_new
            self.prefix_lookup[env_tag] = env_instruction
            
    def _build_env_instruction(self, env_config):
        """Build environment instruction with vocabulary and actions"""
        instruction = env_config.get("env_instruction", "")
        
        # Add grid vocabulary if available
        if env_config.get("grid_vocab"):
            vocab_str = "\nThe meaning of each symbol in the state is:\n" + \
                       ", ".join([f"{k}: {v}" for k, v in env_config["grid_vocab"].items()])
            instruction += vocab_str
            
        # Add action information if available
        if env_config.get("action_lookup"):
            action_str = "\nYour available actions are:\n" + \
                        ", ".join([f"{v}" for k, v in env_config["action_lookup"].items()])
            action_str += f"\nYou can make up to {env_config['max_actions_per_traj']} actions, " + \
                         f"separated by the action separator \" {self.action_sep} \"\n"
            instruction += action_str
            
        return instruction

    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        Convert environment outputs to LLM inputs.
        
        Args:
            env_outputs: List of environment outputs
            prepare_for_update: Whether to prepare for policy update
            
        Returns:
            DataProto containing formatted inputs for LLM
        """
        llm_input_texts = []
        messages_list = []
        
        for env_output in env_outputs:
            # Prepare messages for this environment
            messages = self._prepare_messages(env_output, prepare_for_update)
            messages_list.append(messages)
            
            # Convert messages to text
            text = self._convert_messages_to_text(messages, prepare_for_update)
            llm_input_texts.append(text)
            
        # Tokenize inputs
        inputs = self._tokenize_inputs(llm_input_texts)
        
        # Create DataProto
        llm_inputs = self._create_dataproto(inputs, env_outputs, messages_list, prepare_for_update)
        
        return llm_inputs
        
    def _prepare_messages(self, env_output, prepare_for_update):
        """Prepare messages for a single environment output"""
        messages = [
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
        ]
        
        # Add history
        history = env_output["history"]
        if prepare_for_update and 'state' in history[-1]:
            history = history[:-1]
            
        for idx, content in enumerate(history):
            messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
            
            if "state" in content:
                self._add_state_message(messages, content, env_output["env_id"])
                
            if "llm_response" in content:
                messages.append({"role": "assistant", "content": content["llm_response"]})
                
            if "reward" in content and not (prepare_for_update and idx == len(history) - 1):
                messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                
        return messages
        
    def _add_state_message(self, messages, content, env_id):
        """Add state message to the conversation"""
        format_prompt = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" \
                       if self.config.agent_proxy.enable_think else \
                       "<answer> [your answer] </answer>"
                       
        length_prompt = f"Max response length: {self.env_configs[env_id]['max_tokens']} words (tokens)."
        
        state_message = f"State:\n{content['state']}\n" + \
                       f"You have {content['actions_left']} actions left. " + \
                       f"Always output: {format_prompt} with no extra text. " + \
                       f"Strictly follow this format. {length_prompt}\n"
                       
        messages[-1]["content"] += state_message
        
    def _convert_messages_to_text(self, messages, prepare_for_update):
        """Convert messages to text format"""
        text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=(not prepare_for_update),
            tokenize=False
        )
        
        if not prepare_for_update:
            text += "<think>" if self.config.agent_proxy.enable_think else "<answer>"
            
        return text
        
    def _tokenize_inputs(self, llm_input_texts):
        """Tokenize input texts"""
        return self.tokenizer(
            llm_input_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=False
        )
        
    def _create_dataproto(self, inputs, env_outputs, messages_list, prepare_for_update):
        """Create DataProto object with all necessary information"""
        llm_inputs = DataProto()
        
        # Add tensor data
        llm_inputs.batch = TensorDict({
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "position_ids": inputs.attention_mask.cumsum(dim=-1),
            "responses": inputs.input_ids[:, 1:],
        }, batch_size=inputs.input_ids.shape[0])
        
        # Add non-tensor data
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }
        
        # Add update-specific data
        if prepare_for_update:
            self._add_update_data(llm_inputs, env_outputs, inputs.input_ids)
            
        return llm_inputs
        
    def _add_update_data(self, llm_inputs, env_outputs, input_ids):
        """Add data needed for policy updates"""
        scores = [[i['reward'] for i in env_output['history']] for env_output in env_outputs]
        score_tensor, loss_mask, response_mask = get_masks_and_scores(
            input_ids,
            self.tokenizer,
            scores,
            use_turn_scores=self.config.agent_proxy.use_turn_scores,
            enable_response_mask=self.config.enable_response_mask
        )
        
        # Normalize scores if needed
        if not self.config.agent_proxy.use_turn_scores:
            score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            
        # Add to DataProto
        llm_inputs.batch.update({
            "loss_mask": loss_mask,
            "rm_scores": score_tensor,
            "original_rm_scores": score_tensor,
        })
        
        # Add metrics
        metrics = self._compute_metrics(env_outputs)
        llm_inputs.meta_info = {"metrics": metrics}

    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor
    
    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs

    



@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)
    # batch_list = [
    #     {
    #         "env_ids": 0,
    #         "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
    #     },
    #     {
    #         "env_ids": 1,
    #         "chat_response": "<think> 456. </think><answer> 789 </answer><think> 10123 </think><answer> 11111 </answer>",
    #     }
    # ]
    # ctx_manager.action_sep_lookup = {
    #     0: "|",
    #     1: ";"
    # }
    # for item in batch_list:
    #     item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    # batch_dict = collate_fn(batch_list)
    # batch = DataProto.from_single_dict(batch_dict)
    # env_inputs = ctx_manager.get_env_inputs(batch)
    # print(env_inputs)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 0,
            "metrics": {}
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 1,
            "metrics": {}
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
    
