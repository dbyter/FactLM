# Requirements: pip install torch transformers datasets
"""
FactLM Model Training Script

This script trains a transformer-based language model on Project Gutenberg books 
and UltraChat conversational data. The script is focused purely on training and 
checkpointing - text generation is handled by generate_text.py.

Key features:
- Multi-source data loading via data_loader.py module
- Advanced training with AdamW optimizer and learning rate scheduling
- Automatic checkpointing every 5 epochs with best model tracking
- Early stopping with validation loss monitoring
- Complete model and configuration saving for generate_text.py compatibility
- GPU memory optimization and batch size auto-adjustment

All checkpoints and final models saved by this script are compatible with 
generate_text.py for text generation.
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from factlm_model import FactLM
from data_loader import load_and_process_all_data
import os
import math
from datetime import datetime

def train_model(model, train_data, val_data, epochs, batch_size, sequence_length, learning_rate, device, max_grad_norm=1.0, checkpoint_every=5, save_checkpoints=True):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    
    # Learning rate scheduler with warmup
    steps_per_epoch = len(train_data) // (batch_size * sequence_length)
    
    # Ensure we have enough steps for proper scheduling
    if steps_per_epoch < 1:
        print(f"⚠️  Dataset too small for current batch config!")
        print(f"   Data tokens: {len(train_data):,}")
        print(f"   Tokens per batch: {batch_size * sequence_length:,}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Reducing batch size to make training possible...")
        
        # Auto-adjust batch size to have at least 10 steps per epoch
        max_batch_tokens = len(train_data) // 10
        sequence_length = min(sequence_length, 512)  # Cap sequence length to new max
        batch_size = max(1, max_batch_tokens // sequence_length)
        
        print(f"   New config: batch_size={batch_size}, sequence_length={sequence_length}")
        steps_per_epoch = len(train_data) // (batch_size * sequence_length)
    
    warmup_steps = max(10, steps_per_epoch // 4)  # More warmup steps, at least 10
    total_steps = steps_per_epoch * epochs
    
    print(f"Training schedule: {steps_per_epoch} steps/epoch, {warmup_steps} warmup steps, {total_steps} total steps")
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Checkpoint setup
    checkpoint_dir = None
    if save_checkpoints:
        # Create checkpoints directory with timestamp
        os.makedirs('checkpoints', exist_ok=True)  # Ensure base directory exists
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_UTC')
        checkpoint_dir = os.path.join('checkpoints', f'training_{timestamp}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"💾 Checkpoints will be saved to: {checkpoint_dir}")
        print(f"   Saving every {checkpoint_every} epochs")
    
    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.1, step / warmup_steps)  # Minimum 10% of base LR
        else:
            # Cosine decay after warmup  
            if total_steps <= warmup_steps:
                return 1.0  # No decay if no steps after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(progress * math.pi))
    
    def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, is_best=False):
        """Save a training checkpoint"""
        if not save_checkpoints or checkpoint_dir is None:
            return
            
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'learning_rate': learning_rate,
            'max_grad_norm': max_grad_norm,
            'step': step,
            # Add metadata needed for text generation
            'model_config': {
                'vocab_size': model.embedding.num_embeddings,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'dropout': model.dropout,
                'd_model': model.d_model,
                'max_len': 5000,  # This should match the original config
                'num_heads': model.num_heads
            },
            'timestamp_utc': datetime.now().strftime('%Y%m%d_%H%M%S_UTC'),
            'device_used': str(device),
            'vocab_size': model.embedding.num_embeddings
        }
        
        # Validate checkpoint data for generate_text.py compatibility
        required_fields = ['model_state_dict', 'model_config', 'timestamp_utc', 'device_used', 'vocab_size']
        for field in required_fields:
            if field not in checkpoint_data:
                print(f"⚠️  Warning: Missing required field '{field}' in checkpoint")
        
        # Validate model_config completeness
        required_config_fields = ['vocab_size', 'hidden_size', 'num_layers', 'dropout', 'd_model', 'max_len', 'num_heads']
        for field in required_config_fields:
            if field not in checkpoint_data['model_config']:
                print(f"⚠️  Warning: Missing required model_config field '{field}' in checkpoint")
        
        # Verify saved config matches actual model architecture
        saved_config = checkpoint_data['model_config']
        if (saved_config['d_model'] != model.d_model or 
            saved_config['num_layers'] != model.num_layers or 
            saved_config['num_heads'] != model.num_heads):
            print(f"⚠️  Warning: Saved model_config doesn't match actual model architecture!")
            print(f"   Saved: d_model={saved_config['d_model']}, layers={saved_config['num_layers']}, heads={saved_config['num_heads']}")
            print(f"   Actual: d_model={model.d_model}, layers={model.num_layers}, heads={model.num_heads}")
        
        # Save regular checkpoint
        if epoch % checkpoint_every == 0 or is_best:
            if is_best:
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                print(f"💾 Saving BEST checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
                print(f"💾 Saving checkpoint: {checkpoint_path}")
            
            torch.save(checkpoint_data, checkpoint_path)
        
        # Always save latest checkpoint (for resuming)
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint_data, latest_path)
    
    def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
        """Load a training checkpoint for resuming training"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Return training state information
        resume_info = {
            'epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['best_val_loss'],
            'train_loss': checkpoint['train_loss'],
            'val_loss': checkpoint['val_loss'],
            'step': checkpoint['step']
        }
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Resuming from epoch {resume_info['epoch']}")
        print(f"   Best validation loss: {resume_info['best_val_loss']:.4f}")
        print(f"   Last train loss: {resume_info['train_loss']:.4f}")
        print(f"   Last validation loss: {resume_info['val_loss']:.4f}")
        
        return resume_info
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss since model now outputs raw logits

    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        total_grad_norm = 0
        
        # Create proper sequence batches for better GPU utilization  
        tokens_per_batch = batch_size * sequence_length
        
        for i in range(0, len(train_data) - tokens_per_batch - 1, tokens_per_batch):
            # Get input tokens and target tokens (shifted by 1)
            input_tokens = train_data[i:i + tokens_per_batch]
            target_tokens = train_data[i + 1:i + tokens_per_batch + 1]
            
            if len(input_tokens) < tokens_per_batch or len(target_tokens) < tokens_per_batch:
                continue  # Skip incomplete batches
                
            # Reshape into [batch_size, sequence_length]
            inputs = input_tokens.view(batch_size, sequence_length)
            targets = target_tokens.view(batch_size, sequence_length)

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [batch_size, sequence_length, vocab_size]
            
            # Flatten for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * sequence_length, vocab_size]
            targets = targets.view(-1)  # [batch_size * sequence_length]
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
            total_grad_norm += grad_norm.item()
            
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            total_loss += loss.item()
            num_batches += 1
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Step {step}: Loss {loss.item():.4f}, LR {current_lr:.6f}, Grad Norm {grad_norm.item():.3f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = 0
            
            for i in range(0, len(val_data) - tokens_per_batch - 1, tokens_per_batch):
                input_tokens = val_data[i:i + tokens_per_batch]
                target_tokens = val_data[i + 1:i + tokens_per_batch + 1]
                
                if len(input_tokens) < tokens_per_batch or len(target_tokens) < tokens_per_batch:
                    continue
                    
                inputs = input_tokens.view(batch_size, sequence_length)
                targets = target_tokens.view(batch_size, sequence_length)

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        
        # Early stopping check
        is_best_model = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            is_best_model = True
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}, LR: {current_lr:.6f} ⭐ NEW BEST!")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}, LR: {current_lr:.6f} (patience: {patience_counter}/{patience})")
        
        # Save checkpoints
        save_checkpoint(epoch + 1, model, optimizer, scheduler, avg_train_loss, avg_val_loss, is_best=is_best_model)
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs.")
            break

    return model, batch_size, sequence_length


def save_model(model, tokenizer, model_config, training_stats, device_name):
    """Save the trained model with UTC timestamp and metadata"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate UTC timestamp
    utc_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_UTC')
    model_filename = f"factlm_model_{utc_timestamp}.pth"
    metadata_filename = f"factlm_metadata_{utc_timestamp}.txt"
    
    model_path = os.path.join('models', model_filename)
    metadata_path = os.path.join('models', metadata_filename)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'timestamp_utc': utc_timestamp,
        'device_used': device_name,
        'vocab_size': tokenizer.vocab_size
    }, model_path)
    
    # Save training metadata
    with open(metadata_path, 'w') as f:
        f.write(f"FactLM Model Training Report\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Timestamp (UTC): {utc_timestamp}\n")
        f.write(f"Device Used: {device_name}\n")
        f.write(f"Model File: {model_filename}\n\n")
        
        f.write(f"Model Configuration:\n")
        f.write(f"- Vocabulary Size: {model_config['vocab_size']:,}\n")
        f.write(f"- Model Dimension: {model_config['d_model']}\n")
        f.write(f"- Hidden Size: {model_config['hidden_size']}\n")
        f.write(f"- Number of Layers: {model_config['num_layers']}\n")
        f.write(f"- Number of Heads: {model_config['num_heads']}\n")
        f.write(f"- Head Dimension: {model_config['d_model'] // model_config['num_heads']}\n")
        f.write(f"- Dropout: {model_config['dropout']}\n")
        f.write(f"- Max Length: {model_config['max_len']}\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"- Epochs: {training_stats['epochs']}\n")
        f.write(f"- Batch Size: {training_stats['batch_size']}\n")
        f.write(f"- Sequence Length: {training_stats['sequence_length']}\n")
        f.write(f"- Effective Batch Size: {training_stats['batch_size'] * training_stats['sequence_length']:,} tokens\n")
        f.write(f"- Learning Rate: {training_stats['learning_rate']}\n")
        f.write(f"- Optimizer: AdamW with weight decay 0.1, betas=(0.9, 0.95)\n")
        f.write(f"- LR Schedule: Extended Warmup + Cosine Decay\n")
        f.write(f"- Early Stopping: Patience=5 epochs\n")
        f.write(f"- Gradient Clipping: {training_stats['max_grad_norm']}\n")
        f.write(f"- Training Tokens: {training_stats['train_tokens']:,}\n")
        f.write(f"- Validation Tokens: {training_stats['val_tokens']:,}\n")
        f.write(f"- Total Parameters: {training_stats['total_params']:,}\n\n")
        
        if 'book_files' in training_stats:
            f.write(f"Training Data:\n")
            for book_file in training_stats['book_files']:
                f.write(f"- {book_file}\n")
            
            if 'ultrachat_tokens' in training_stats and training_stats['ultrachat_tokens'] > 0:
                f.write(f"\nUltraChat Data:\n")
                f.write(f"- Dataset: stingning/ultrachat\n")
                f.write(f"- Conversations: {training_stats.get('ultrachat_conversations', 0):,}\n")
                f.write(f"- Tokens: {training_stats['ultrachat_tokens']:,}\n")
            
            if 'wikipedia_tokens' in training_stats and training_stats['wikipedia_tokens'] > 0:
                f.write(f"\nWikipedia Data:\n")
                f.write(f"- Dataset: wikimedia/wikipedia (20231101.en)\n")
                f.write(f"- Articles: {training_stats.get('wikipedia_articles', 0):,}\n")
                f.write(f"- Tokens: {training_stats['wikipedia_tokens']:,}\n")
            
            # Data mix summary
            if 'ultrachat_tokens' in training_stats or 'wikipedia_tokens' in training_stats:
                total_tokens = (training_stats.get('book_tokens', 0) + 
                               training_stats.get('ultrachat_tokens', 0) + 
                               training_stats.get('wikipedia_tokens', 0))
                if total_tokens > 0:
                    book_pct = training_stats.get('book_tokens', 0) / total_tokens
                    ultrachat_pct = training_stats.get('ultrachat_tokens', 0) / total_tokens
                    wiki_pct = training_stats.get('wikipedia_tokens', 0) / total_tokens
                    f.write(f"\nData Mix:\n")
                    f.write(f"- Books: {book_pct:.1%} ({training_stats.get('book_tokens', 0):,} tokens)\n")
                    f.write(f"- UltraChat: {ultrachat_pct:.1%} ({training_stats.get('ultrachat_tokens', 0):,} tokens)\n")
                    f.write(f"- Wikipedia: {wiki_pct:.1%} ({training_stats.get('wikipedia_tokens', 0):,} tokens)\n")
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    
    return model_path, metadata_path


# Example usage and data setup
if __name__ == "__main__":
    print("🚀 FactLM Training - Efficient Model")
    print("=" * 50)
    print("📝 Features: Smaller model (d_model=256), 512-token sequences, automatic checkpointing")
    print("📈 Dataset: Books + 15K UltraChat + Generated data + 25K Wikipedia")
    print("💾 Checkpoints saved to: checkpoints/training_TIMESTAMP/")
    print("🔄 Resume training by modifying this script to load from checkpoint")
    
    # Load and process all data using the data_loader module
    training_data, validation_data, data_stats = load_and_process_all_data(
        data_dir='data',
        ultrachat_samples=15000,  # 15K UltraChat conversations (reduced from 50K)
        wikipedia_samples=25000,  # 25K Wikipedia articles (reduced from 50K)
        train_split=0.8,
        seed=42
    )
    
    # Extract data statistics for training metadata
    tokenizer = data_stats['tokenizer']
    book_files = data_stats['book_files']
    
    print(f"\n📊 Final dataset statistics:")
    print(f"Training data shape: {training_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Sample tokens: {training_data[:20].tolist()}")
    print(f"Sample decoded: {tokenizer.decode(training_data[:50])}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    # Smaller, more efficient model configuration
    # NOTE: This configuration must match the defaults in generate_text.py for checkpoint compatibility
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 256,      # Match d_model for efficiency
        'num_layers': 6,         # Reduced from 12 to lower parameter count
        'dropout': 0.2,          # Standard dropout
        'd_model': 256,          # Reduced from 512 for efficiency
        'max_len': 5000,
        'num_heads': 8           # 8 heads gives 32-dim heads (256/8=32)
    }
    
    # Validate model configuration
    assert model_config['d_model'] % model_config['num_heads'] == 0, \
        f"d_model ({model_config['d_model']}) must be divisible by num_heads ({model_config['num_heads']})"
    
    head_dim = model_config['d_model'] // model_config['num_heads']
    print(f"Model architecture validated: {model_config['num_heads']} heads × {head_dim} dimensions = {model_config['d_model']}")
    
    model = FactLM(**model_config)
    
    # Training parameters - automatically select best available device
    device = None
    device_name = ""
    
    # Try CUDA first
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            device_name = f"CUDA (NVIDIA GPU: {torch.cuda.get_device_name()})"
            print(f"CUDA detected: {torch.cuda.device_count()} GPU(s) available")
        except Exception as e:
            print(f"CUDA available but failed to initialize: {e}")
            device = None
    
    # Try MPS second (only if CUDA failed)
    if device is None:
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                device_name = "MPS (Apple Silicon)"
                print("MPS detected and available")
        except Exception as e:
            print(f"MPS check failed: {e}")
            device = None
    
    # Fallback to CPU
    if device is None:
        device = torch.device("cpu")
        device_name = "CPU"
        print("Using CPU as fallback")
    
    epochs = 25           # Keep same epochs for good training
    batch_size = 24       # Reduced due to doubled sequence length (was 48)
    sequence_length = 512  # Increased from 256 for better long-context learning
    learning_rate = 0.0003 # Slightly increased LR for smaller model
    max_grad_norm = 1.0   # Gradient clipping
    checkpoint_every = 5  # Save checkpoint every 5 epochs
    
    print(f"\n⚙️  Training configuration:")
    print(f"Device: {device} ({device_name})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch configuration: {batch_size} sequences × {sequence_length} tokens = {batch_size * sequence_length:,} tokens per batch")
    print(f"🚀 LONG CONTEXT: Doubled sequence length to {sequence_length} tokens for better conversation understanding")
    print(f"Checkpoint frequency: Every {checkpoint_every} epochs")
    print(f"💡 Head dimension: {head_dim} (optimized for efficiency)")
    
    # Check if configuration is reasonable for dataset size
    total_tokens_needed = batch_size * sequence_length
    if total_tokens_needed > len(training_data) // 2:
        print(f"⚠️  Batch size may be too large for dataset ({total_tokens_needed:,} tokens per batch vs {len(training_data):,} total tokens)")
        print("   The training function will auto-adjust if needed...")
        print(f"   Current config: {batch_size} × {sequence_length} = {total_tokens_needed:,} tokens per batch")
    
    estimated_memory_mb = (batch_size * sequence_length * model_config['d_model'] * 4) // (1024**2)
    print(f"Estimated GPU memory usage: ~{estimated_memory_mb}MB per batch")
    
    if estimated_memory_mb > 6000:  # More than 6GB per batch (increased due to longer sequences)
        print("⚠️  High memory usage detected! Monitor GPU memory during training")
        print(f"   Consider reducing batch_size if you get OOM errors")
    else:
        print(f"✅ Memory usage looks reasonable: ~{estimated_memory_mb}MB per batch (longer sequences)")
    
    # Train the model
    print("\n🎯 Starting training...")
    
    # Collect training statistics
    training_stats = {
        'epochs': epochs,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'learning_rate': learning_rate,
        'max_grad_norm': max_grad_norm,
        'train_tokens': len(training_data),
        'val_tokens': len(validation_data),
        'total_params': sum(p.numel() for p in model.parameters()),
        'book_files': data_stats['book_files'],
        'book_tokens': data_stats['book_tokens'],
        'ultrachat_tokens': data_stats['ultrachat_tokens'],
        'ultrachat_conversations': data_stats['ultrachat_conversations'],
        'wikipedia_tokens': data_stats['wikipedia_tokens'],
        'wikipedia_articles': data_stats['wikipedia_articles']
    }
    
    trained_model, final_batch_size, final_sequence_length = train_model(
        model=model,
        train_data=training_data,
        val_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        device=device,
        max_grad_norm=max_grad_norm,
        checkpoint_every=checkpoint_every,
        save_checkpoints=True
    )
    
    # Update training stats with final batch configuration
    training_stats['batch_size'] = final_batch_size
    training_stats['sequence_length'] = final_sequence_length
    
    # Save the trained model
    print("\n💾 Saving trained model...")
    model_path, metadata_path = save_model(
        model=trained_model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_stats=training_stats,
        device_name=device_name
    )
    
    # Training complete! 
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved: {model_path}")
    print(f"📊 Metadata saved: {metadata_path}")
    print(f"\n🚀 To generate text with this model, run:")
    print(f"   python generate_text.py {model_path}")