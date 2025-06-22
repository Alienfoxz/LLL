from peft import PeftConfig, get_peft_model, get_peft_config
from typing import Optional, Mapping, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, Seq2SeqTrainingArguments, GenerationConfig, Seq2SeqTrainer, DataCollatorForSeq2Seq, EvalPrediction
import torch
from datasets import Dataset, load_dataset
import functools
import dataclasses
import ruamel.yaml as yaml
from typing import Optional, Union
from pathlib import Path
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import os


@dataclasses.dataclass
class DataConfig(object):
    data_dir: Optional[str] = None
    file_name: Optional[str] = None
    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix



@dataclasses.dataclass
class FinetuningConfig(object):
    data_config: DataConfig
    max_length: int
    do_lora : bool
    
    training_args: Seq2SeqTrainingArguments = dataclasses.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None
    def __post_init__(self):
        if not self.training_args.do_eval:
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)
        
        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(config_dict=peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = Path(path)
        parser = yaml.YAML(typ='safe', pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)

    

def load_tokenizer_and_model(
        model_name: str,
        peft_config: Optional[PeftConfig] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16  # Must use BFloat 16
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16
        )
    return tokenizer, model

    
def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        combine: bool,
) -> dict[str, list]:
    vocab = tokenizer.get_vocab()
    batched_conv = batch['messages']
    batched_input_ids = []
    batched_labels = []
    for conv in batched_conv:
        # 找到最后一个助手的回复开始位置,即user之后
        '''
        sample:
            <|im_start|>system
            Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
            <|im_start|>user
            Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$.<|im_end|>
            <|im_start|>assistant
        '''
        im_start_token_id = vocab.get('<|im_start|>')
        im_end_token_id = vocab.get('<|im_end|>')
        # system_token_id = vocab.get('system')
        assistant_token_id = vocab.get('assistant')

        # 将对话转换成单一序列
        if combine:
            input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
            loss_masks = [False] * len(input_ids)
            # 找到assistant回复的content的模板
            last_assistant_start_index = -1
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == im_start_token_id and i + 1 < len(input_ids) and input_ids[i+1] == assistant_token_id:
                    last_assistant_start_index = i + 2 # 跳过 <|im_start|> 和 assistant token
                    break
                
            if last_assistant_start_index != -1: # 如果无assistant
                    # 从助手回复内容开始计算损失，直到 <|im_end|>
                    for j in range(last_assistant_start_index, len(input_ids)):
                        if input_ids[j] == im_end_token_id: # 遇到 <|im_end|> 停止计算损失
                            break
                        loss_masks[j] = True
                        
        # 逐条处理
        else:
            input_ids = []
            for message in conv:
                # 根据消息角色决定是否计算损失
                loss_mask_val = False if message['role'] in ('system', 'user') else True
                
                new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False) # 转换成token
                input_ids += new_input_ids
                loss_masks += [loss_mask_val] * len(new_input_ids)
            del new_input_ids, loss_mask_val

        # 由于是自回归，所以第一个是不需要预测的
        loss_masks = [False, *loss_masks[:-1]]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100) # mask的部分即代表label为-100不需要预测   
        
        # 设置最大长度
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    
    
    del batched_conv, conv, input_ids, loss_masks, labels
    torch.cuda.empty_cache()

    return {'input_ids': batched_input_ids, 'labels': batched_labels}

def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    batched_pred_ids, batched_label_ids = eval_preds
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3))
    return {k: np.mean(v) for k, v in metrics_dct.items()}


def main(
    model_name,
    config_file_path,
    auto_resume_from_checkpoint: Optional[None | int] = None,
):
    ft_config = FinetuningConfig.from_file(config_file_path)
    print(ft_config)
    if ft_config.do_lora:
        tokenizer, model = load_tokenizer_and_model(model_name, peft_config=ft_config.peft_config)
    else:
        tokenizer, model = load_tokenizer_and_model(model_name, peft_config=None)
    dataset = load_dataset(ft_config.data_config.data_dir, data_files=ft_config.data_config.file_name, split="train", num_proc=ft_config.data_config.num_proc)
    dataset = dataset.map(functools.partial(
            process_batch,
            tokenizer=tokenizer,
            combine=True,
            max_length=2048,
        ), batched=True, num_proc=4, remove_columns=dataset.column_names)
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=dataset,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )
    # with open("4.txt", "w", encoding="utf-8") as f:
    #     f.write(str(model))
        # for n,p in model.named_parameters():
        #     f.write(f"name: {n}, grad: {p.requires_grad}\n")
    if auto_resume_from_checkpoint: # 调用之前的检查点继续训练
        if auto_resume_from_checkpoint == -1: # 选择checkpointnum最大的检查点
            dirlist = os.listdir(ft_config.training_args.output_dir)
            checkpoint_sn = 0
            for checkpoint_str in dirlist:
                if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                    checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                    if checkpoint > checkpoint_sn:
                        checkpoint_sn = checkpoint
                        checkpoint_directory = os.path.join(ft_config.training_args.output_dir, "checkpoint-" + str(checkpoint))
        else:
            checkpoint_directory = os.path.join(ft_config.training_args.output_dir, "checkpoint-" + str(auto_resume_from_checkpoint))
        trainer.train(resume_from_checkpoint=checkpoint_directory)
    else:
        trainer.train()
    
if __name__ == "__main__":
    main(model_name="Qwen/Qwen2.5-Math-7B-Instruct", config_file_path="./configs/config.yaml")