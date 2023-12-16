from datasets import load_dataset
from transformers import LongformerTokenizer
from transformers import TrainingArguments, Trainer
import evaluate
from transformers import DataCollatorWithPadding
import numpy as np
import torch
from transformers import LongformerForSequenceClassification, LongformerSelfAttention
from tqdm import tqdm
import numpy as np
import time
import argparse
import os.path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import json
# from gp_apis import gp_mm, gp_sum_two_tensors, gp_transpose
import math
import os
import random 

print('mickey mouse')
seed = 42
torch.cuda.manual_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda")

class LinearFunction(torch.autograd.Function):
    """
    References:
    - PyTorch Extending PyTorch: https://pytorch.org/docs/stable/notes/extending.html
    - Custom C++ and CUDA Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html#integrating-a-c-cuda-operation-with-pytorch
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        dim0, dim1 = input.size(0), weight.size(0)
        weightT = gp_transpose(weight, weight.size(1), weight.size(0), device)
        output = gp_mm(input, weightT, dim0, dim1, device)
        if bias is not None:
            dim0, dim1 = output.size(0), output.size(1)
            output = gp_sum_two_tensors(output, bias.unsqueeze(0).expand_as(output), dim0, dim1, device)
        return output

    @staticmethod
    def backward(ctx, dZ):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            dim_0, dim_1 = input.shape
#             weightT = gp_transpose(weight, weight.size(1), weight.size(0), device)
            grad_input = gp_mm(dZ, weight, dim_0, dim_1, device)
            
        if ctx.needs_input_grad[1]:
            dim_0, dim_1 = dZ.shape
            dZT = gp_transpose(dZT, dZT.size(1), dZT.size(0), device)
            grad_weight = gp_mm(dZT, input, dim_0, dim_1, device)
            
        if bias is not None and ctx.needs_input_grad[2]:
#             dim0, dim1 = bias.size(0), dZ.size(0)
#             ones = torch.ones(1, dim1).to(device)
            grad_bias = dZ.sum(0)#gp_mm(ones, dZ, 1, dim0, device).squeeze()

        return grad_input, grad_weight, grad_bias

    
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.input_features = in_features
        self.output_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

print('mickey mouse')
class CustomLongformerForSequenceClassification(LongformerForSequenceClassification):
    def __init__(self, config):
        super(CustomLongformerForSequenceClassification, self).__init__(config)
        for i, layer in enumerate(self.longformer.encoder.layer):
            layer.attention.self = CustomLongformerSelfAttention(config, layer_id=i)


class CustomLongformerSelfAttention(LongformerSelfAttention):
    def __init__(self, config, layer_id):
        super(CustomLongformerSelfAttention, self).__init__(config, layer_id)
        self.query = CustomLinear(config.hidden_size, self.embed_dim)
        self.key = CustomLinear(config.hidden_size, self.embed_dim)
        self.value = CustomLinear(config.hidden_size, self.embed_dim)

        self.query_global = CustomLinear(config.hidden_size, self.embed_dim)
        self.key_global = CustomLinear(config.hidden_size, self.embed_dim)
        self.value_global = CustomLinear(config.hidden_size, self.embed_dim)
        
        

imdb = load_dataset("imdb")

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096", num_labels=2, id2label=id2label, label2id=label2id
)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_imdb = imdb.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

training_args = TrainingArguments(
    output_dir="custom_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=1000,
    save_steps=1000,
    save_strategy="steps",
    load_best_model_at_end=True,
    gradient_accumulation_steps=2, 
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()