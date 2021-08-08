from Project import Project
from data import get_sugar_lyrics
from transformers import GPT2Tokenizer, GPT2Model, Trainer, \
    TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling, \
    AutoModelForCausalLM
from datasets import load_metric
from torch.utils.data import random_split

import argparse
import torch
import numpy as np


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def tokenize_function(example, args):
    return tokenizer(example, max_length = args.max_length, truncation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='test set size', metavar="[0-1]")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_train_epochs", type=int, default=4,
                        help="number of epochs for the training loop")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                        help="The batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                        help="The batch size per GPU/TPU core/CPU for "
                             "evaluation")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of steps used for a linear warmup from "
                             "0 to learning_rate")
    parser.add_argument("--max_length", type=int, default=768,
                        help="The maximum length (in number of tokens) for "
                             "the inputs to the transformer model. ")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="The weight decay to apply (if not zero) to "
                             "all layers except all bias and LayerNorm weights in AdamW optimizer")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(args.device)
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    project = Project()
    # Get the data and store it in the dataset_dir  as .txt files. Already
    # split into train and test sets
    get_sugar_lyrics(project.dataset_dir)

    # Instantiate italian GPT2 tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the data.
    with open(project.dataset_dir / 'sugar_lyrics.txt', encoding="utf8") as f:
        songs = [s.strip('\n') for s in f.readlines()]

    # Tokenize the text.
    tokenized_dataset = [tokenize_function(song, args) for song in
                         songs]

    # Split into training and validation sets.
    val_size = int(args.test_size * len(tokenized_dataset))
    train_size = len(tokenized_dataset) - val_size
    train_dataset, val_dataset = random_split(tokenized_dataset,
                                              [train_size, val_size])

    # Form batches by applying padding and also random data augmentation (
    # like random masking).
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)
    # Fine-tune the model using the 🤗 Trainer API
    model = AutoModelForCausalLM.from_pretrained("GroNLP/gpt2-small-italian")

    training_args = TrainingArguments(
        output_dir=project.result_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=project.log_dir,
        evaluation_strategy='epoch'
    )

    metric = load_metric("bleu")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()

