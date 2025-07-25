#!/usr/bin/env python3
"""
FactLM Model Fine-tuner

This script fine-tunes an existing trained FactLM model using generated conversational data.
Perfect for taking a model trained on factual knowledge (books + Wikipedia) and adding
conversational abilities using GPT-4o generated training data.

Usage:
    python model_fine_tuner.py <model_checkpoint_path>

Features:
- Loads existing model from checkpoint
- Fine-tunes using generated conversational data only
- Conservative hyperparameters to preserve existing knowledge
- Automatic checkpointing and validation
- Compatible with generate_text.py for testing

Example:
    python model_fine_tuner.py checkpoints/training_20250101_120000_UTC/best_model.pth
"""

# Fix for multiprocessing issues on macOS/forked processes
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from factlm_model import FactLM
import json
import sys
import math
from datetime import datetime


class ConversationDataset(Dataset):
    """Dataset for conversational fine-tuning"""
    
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"🔄 Processing {len(conversations)} conversations for fine-tuning...")
        
        for conv in conversations:
            if 'data' in conv and len(conv['data']) == 2:
                prompt, response = conv['data']
                
                # Format as a conversation
                conversation_text = f"Human: {prompt}\n\nAssistant: {response}"
                
                # Tokenize
                tokens = tokenizer.encode(conversation_text, add_special_tokens=True, max_length=max_length, truncation=True)
                
                if len(tokens) > 10:  # Skip very short conversations
                    self.samples.append(torch.tensor(tokens, dtype=torch.long))
        
        print(f"✅ Prepared {len(self.samples)} conversation samples for fine-tuning")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Input is all tokens except the last one
        inputs = tokens[:-1]
        # Target is all tokens except the first one
        targets = tokens[1:]
        
        return inputs, targets


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    inputs, targets = zip(*batch)
    
    # Find max length in this batch
    max_len = max(len(seq) for seq in inputs)
    
    # Pad sequences
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        # Pad with zeros (we'll mask these in loss calculation)
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
        
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print("✅ Found model configuration in checkpoint")
    else:
        # Reconstruct config from model state dict
        print("⚠️  No model config found, reconstructing from state dict...")
        model_state = checkpoint['model_state_dict']
        
        vocab_size = model_state['embedding.weight'].shape[0]
        d_model = model_state['embedding.weight'].shape[1]
        
        # Count layers
        num_layers = 0
        for key in model_state.keys():
            if key.startswith('encoder.layers.') and '.self_attn.in_proj_weight' in key:
                layer_num = int(key.split('.')[2])
                num_layers = max(num_layers, layer_num + 1)
        
        model_config = {
            'vocab_size': vocab_size,
            'hidden_size': d_model,
            'num_layers': num_layers,
            'dropout': 0.2,
            'd_model': d_model,
            'max_len': 5000,
            'num_heads': 8
        }
        print(f"   Reconstructed: {model_config}")
    
    # Initialize model
    model = FactLM(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Architecture: {model_config['num_layers']} layers, {model_config['d_model']} dim, {model_config['num_heads']} heads")
    
    return model, model_config


def load_generated_conversations(data_file="generated_training_data.json"):
    """Load generated conversational data"""
    print(f"📊 Loading generated conversations from: {data_file}")
    
    if not os.path.exists(data_file):
        # Try alternative file names
        alternatives = ["temp_generated_training_data.json", "generated_training_data_raw.json"]
        for alt in alternatives:
            if os.path.exists(alt):
                data_file = alt
                print(f"   Found alternative file: {alt}")
                break
        else:
            raise FileNotFoundError(f"No generated training data found. Tried: {data_file}, {alternatives}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    print(f"✅ Loaded {len(conversations)} conversations")
    
    # Validate and filter conversations
    valid_conversations = []
    for i, conv in enumerate(conversations):
        if 'data' in conv and len(conv['data']) == 2:
            prompt, response = conv['data']
            if len(prompt.strip()) > 10 and len(response.strip()) > 10:
                valid_conversations.append(conv)
    
    print(f"✅ Validated {len(valid_conversations)} usable conversations")
    
    if len(valid_conversations) == 0:
        raise ValueError("No valid conversations found in the data file")
    
    return valid_conversations


def fine_tune_model(model, conversations, tokenizer, device, epochs=3, batch_size=8, learning_rate=1e-5, max_length=512):
    """Fine-tune the model on conversational data"""
    
    print(f"\n🎯 Starting fine-tuning...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max sequence length: {max_length}")
    
    # Create dataset and dataloader
    dataset = ConversationDataset(conversations, tokenizer, max_length)
    
    # Split into train/val (90/10 for fine-tuning)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Single-threaded to avoid issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"✅ Created DataLoaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Conservative optimizer for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch // 2  # Half epoch warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(progress * math.pi)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # Ignore padding tokens
    
    print(f"📈 Training schedule: {steps_per_epoch} steps/epoch, {warmup_steps} warmup steps")
    
    # Training loop
    best_val_loss = float('inf')
    step = 0
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_UTC')
    checkpoint_dir = os.path.join('fine_tuned', f'fine_tuning_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"💾 Fine-tuned models will be saved to: {checkpoint_dir}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            step += 1
            
            if step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Step {step}: Loss {loss.item():.4f}, LR {current_lr:.6f}, Grad Norm {grad_norm.item():.3f} [{progress:.1f}%]")
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f} ⭐ NEW BEST!")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'step': step,
            'fine_tuning': True,
            'conversations_used': len(conversations)
        }
        
        # Save epoch checkpoint
        epoch_path = os.path.join(checkpoint_dir, f'fine_tuned_epoch_{epoch+1:02d}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"💾 Saved: {epoch_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'fine_tuned_best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved BEST: {best_path}")
    
    print(f"\n🎉 Fine-tuning completed!")
    print(f"📁 Fine-tuned models saved to: {checkpoint_dir}")
    return model, checkpoint_dir


def main():
    if len(sys.argv) != 2:
        print("Usage: python model_fine_tuner.py <model_checkpoint_path>")
        print("\nExample:")
        print("  python model_fine_tuner.py checkpoints/training_20250101_120000_UTC/best_model.pth")
        print("\nThis will fine-tune the model using generated conversational data.")
        return
    
    checkpoint_path = sys.argv[1]
    
    print("🚀 FactLM Model Fine-tuner")
    print("=" * 50)
    print("🎯 Fine-tuning existing model with conversational data")
    print("📊 Using generated GPT-4o training data")
    print("🔧 Conservative hyperparameters to preserve knowledge")
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA (NVIDIA GPU: {torch.cuda.get_device_name()})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"Using device: {device} ({device_name})")
    
    try:
        # Load tokenizer
        print("\n🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model, model_config = load_model_from_checkpoint(checkpoint_path, device)
        
        # Load conversational data
        conversations = load_generated_conversations()
        
        # Fine-tune the model
        fine_tuned_model, save_dir = fine_tune_model(
            model=model,
            conversations=conversations,
            tokenizer=tokenizer,
            device=device,
            epochs=3,  # Conservative number of epochs
            batch_size=8,  # Smaller batch size for fine-tuning
            learning_rate=1e-5,  # Low learning rate to preserve existing knowledge
            max_length=512
        )
        
        print(f"\n🚀 To test your fine-tuned model, run:")
        print(f"   python generate_text.py {save_dir}/fine_tuned_best.pth")
        
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 