from datasets import load_from_disk
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
from transformers import Blip2Config, Blip2VisionConfig, Blip2ForConditionalGeneration, Blip2QFormerConfig
from transformers import AutoModel, AutoTokenizer, SwinModel, SwinConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Blip2Processor, AutoImageProcessor, BlipImageProcessor
from transformers import AddedToken
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    decoded_preds = processor.tokenizer.batch_decode(logits, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
    # We can return ROUGE-1, ROUGE-2, and ROUGE-L as needed
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        question = item['input_ids']
        answer = self.dataset[idx]['labels']
        image = item["pixel_values"]
        text = question
        
        encoding = self.processor(image, text, padding="max_length", max_length= 512, truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length= 512, pad_to_max_length=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding

##################################  Model Creation  ##################################
llm_id = "meta-llama/Llama-3.2-3B"
vis_id = "microsoft/swin-tiny-patch4-window7-224"

vision_config = Blip2VisionConfig(
    model_type="swin",
    hidden_size=768,  # Match Swin's hidden size
    num_hidden_layers=4,
    num_attention_heads=12,
    image_size=224,
    patch_size=4
)

llama_model = AutoModelForCausalLM.from_pretrained(llm_id)

qformer_config = Blip2QFormerConfig(
    vocab_size=llama_model.config.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,  # Typically set for feed-forward layers
    cross_attention_frequency=2,  # Frequency of cross-attention layers
    encoder_hidden_size=vision_config.hidden_size  # Match Swin Transformer output hidden size
)

blip2_config = Blip2Config.from_vision_qformer_text_configs(
    vision_config=vision_config,
    qformer_config=qformer_config,  # Configure this as per requirements
    text_config=llama_model.config  # LLaMA configuration
)

model = Blip2ForConditionalGeneration(config=blip2_config)
model.vision_model = SwinModel.from_pretrained(vis_id)
model.language_model = llama_model

image_processor = BlipImageProcessor.from_pretrained(vis_id)
llama_tokenizer = AutoTokenizer.from_pretrained(llm_id)
processor = Blip2Processor(image_processor, llama_tokenizer)

processor.num_query_tokens = model.config.num_query_tokens
image_token = AddedToken("<image>", normalized=False, special=True)
processor.tokenizer.add_tokens([image_token], special_tokens=True)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64,)
model.config.image_token_index = len(processor.tokenizer) - 1

for param in model.vision_model.parameters():
    param.requires_grad = False

for param in model.language_model.parameters():
    param.requires_grad = False

for param in model.qformer.parameters():
    param.requires_grad = True

##################################      Creating Dataset and Dataloader     ##################################
dataset = load_from_disk("Path to dataset")
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.rename_column("image", "pixel_values")
dataset = dataset.rename_column("question", "input_ids")
dataset = dataset.rename_column("answer", "labels")
dataset = dataset.remove_columns(["class"])

train_dataset = VQADataset(dataset=dataset["train"],
                          processor=processor)
valid_dataset = VQADataset(dataset=dataset["test"],
                          processor=processor)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)

##################################      Training Arguements     ##################################
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

num_epochs = 100
patience = 10
min_eval_acc = float("-inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

##################################      Model Training and Evaluation      ##################################
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    train_acc = 0
    train_tqdm = tqdm(range(len(train_dataloader)), desc=f'Epoch {epoch+1} - Training loss: 0.000 - Train Acc: 0.000', position=0)
    for idx, batch in zip(train_tqdm, train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        # attention_mask=attention_masked,
                        labels=labels)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        logits = outputs.logits.argmax(dim=-1)
        
        train_acc += compute_metrics((logits, labels))["rougeL"]  # You can use rouge1, rouge2, or rougeL
        
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_tqdm.set_description(f'Epoch {epoch+1} - Training loss: {epoch_loss/(idx+1):.4f} - Train Acc: {train_acc/(idx+1):.4f}')
        # Clear cache to avoid OOM
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    model.eval()
    eval_loss = 0
    eval_acc = 0
    val_tqdm = tqdm(range(len(valid_dataloader)), desc=f'Epoch {epoch+1} - Eval loss: 0.000 - Eval Acc: 0.000')
    for idx, batch in zip(val_tqdm, valid_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_masked,
                        labels=labels)
        
        loss = outputs.loss
        eval_loss += loss.item()

        logits = outputs.logits.argmax(dim=-1)
        eval_acc += compute_metrics((logits, labels))["rougeL"]  # You can use rouge1, rouge2, or rougeL

        val_tqdm.set_description(f'Epoch {epoch+1} - Eval loss: {eval_loss/(idx+1):.4f} - Eval Acc: {eval_acc/(idx+1):.4f}')
        # Clear cache to avoid OOM
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    tracking_information.append((epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch+1, epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    scheduler.step()
    if eval_acc > min_eval_acc:
        model.save_pretrained("/scratch/rdinesh2/Agro_project/models/blip2_pt", from_pt=True) 
        print("Saved model")
        min_eval_acc = eval_acc
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break
    
pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")
