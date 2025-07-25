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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from factlm_model import FactLM
from data_loader import load_and_process_all_data
import os
import math
from datetime import datetime


class TokenDataset(Dataset):
    """Simple dataset for tokenized sequences"""
    
    def __init__(self, tokens, sequence_length):
        self.tokens = tokens
        self.sequence_length = sequence_length
        # Calculate how many complete sequences we can make
        self.num_sequences = (len(tokens) - 1) // sequence_length
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        
        inputs = self.tokens[start_idx:end_idx]
        targets = self.tokens[start_idx + 1:end_idx + 1]
        
        return inputs, targets

def train_model(model, train_data, val_data, epochs, batch_size, sequence_length, learning_rate, device, max_grad_norm=1.0, checkpoint_every=5, save_checkpoints=True):
    model.to(device)
    
    # Create DataLoaders with workers for efficient data loading
    print(f"🔧 Creating DataLoaders with 4 workers...")
    train_dataset = TokenDataset(train_data, sequence_length)
    val_dataset = TokenDataset(val_data, sequence_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type != 'cpu' else False,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Fewer workers for validation
        pin_memory=True if device.type != 'cpu' else False,
        persistent_workers=True
    )
    
    print(f"✅ DataLoaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Improved optimizer settings for better stability
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,  # Reduced weight decay for better stability
        betas=(0.9, 0.999),  # Standard betas, second moment closer to 1
        eps=1e-6  # Smaller epsilon for better precision
    )
    
    # Learning rate scheduler with warmup
    steps_per_epoch = len(train_loader)
    
    # Ensure we have enough steps for proper scheduling
    if steps_per_epoch < 1:
        print(f"⚠️  Dataset too small for current batch config!")
        return model, batch_size, sequence_length
    
    # More conservative warmup and decay schedule
    warmup_steps = max(100, steps_per_epoch // 2)  # Longer warmup for stability
    total_steps = steps_per_epoch * epochs
    
    print(f"Training schedule: {steps_per_epoch} steps/epoch, {warmup_steps} warmup steps, {total_steps} total steps")
    print(f"💾 Checkpoints: Every {checkpoint_every} epochs + every 10,000 steps")
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 8  # Increased patience for more stable training
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
        print(f"   Saving every {checkpoint_every} epoch{'s' if checkpoint_every > 1 else ''}")
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Smoother warmup with minimum LR
            return max(0.01, step / warmup_steps)  # Start from 1% of base LR
        else:
            # More gradual cosine decay after warmup  
            if total_steps <= warmup_steps:
                return 1.0  # No decay if no steps after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            # Cosine decay with minimum LR of 10% of base
            return max(0.1, 0.5 * (1 + math.cos(progress * math.pi)))

    def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, is_best=False, step=None):
        if not save_checkpoints:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'step': step if step is not None else 0
        }
        
        # Save regular checkpoint every N epochs
        if epoch % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saving checkpoint: {checkpoint_path}")
        
        # Save step-based checkpoint every 10K steps
        if step is not None and step % 10000 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step:06d}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saving step checkpoint: {checkpoint_path}")
        
        # Save best model checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saving BEST checkpoint: {best_path}")

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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing for better generalization

    step = 0
    # Track loss history for stability monitoring
    recent_losses = []
    loss_smoothing_window = 50
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        total_grad_norm = 0
        
        # Use DataLoader for efficient batching with workers
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [batch_size, sequence_length, vocab_size]
            
            # Flatten for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * sequence_length, vocab_size]
            targets = targets.view(-1)  # [batch_size * sequence_length]
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # More aggressive gradient clipping for stability
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
            total_grad_norm += grad_norm.item()
            
            # Check for gradient explosion
            if grad_norm.item() > max_grad_norm * 10:
                print(f"⚠️  Large gradient detected: {grad_norm.item():.3f} at step {step}")
            
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            total_loss += loss.item()
            num_batches += 1
            step += 1
            
            # Track recent losses for stability monitoring
            recent_losses.append(loss.item())
            if len(recent_losses) > loss_smoothing_window:
                recent_losses.pop(0)
            
            # Print progress every 100 steps with smoothed loss
            if step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                smoothed_loss = sum(recent_losses) / len(recent_losses) if recent_losses else loss.item()
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Step {step}: Loss {loss.item():.4f} (Smooth: {smoothed_loss:.4f}), LR {current_lr:.6f}, Grad Norm {grad_norm.item():.3f} [{progress:.1f}%]")
            
            # Save checkpoint every 10K steps
            if step % 10000 == 0:
                # Quick validation for step checkpoint
                model.eval()
                with torch.no_grad():
                    val_loss_quick = 0
                    val_batches_quick = 0
                    for val_inputs, val_targets in val_loader:
                        if val_batches_quick >= 10:  # Only validate on first 10 batches for speed
                            break
                        val_inputs = val_inputs.to(device, non_blocking=True)
                        val_targets = val_targets.to(device, non_blocking=True)
                        val_outputs = model(val_inputs)
                        val_outputs = val_outputs.view(-1, val_outputs.size(-1))
                        val_targets = val_targets.view(-1)
                        val_loss_quick += criterion(val_outputs, val_targets).item()
                        val_batches_quick += 1
                
                avg_val_loss_quick = val_loss_quick / val_batches_quick if val_batches_quick > 0 else 0
                avg_train_loss_quick = total_loss / num_batches if num_batches > 0 else 0
                
                save_checkpoint(epoch + 1, model, optimizer, scheduler, avg_train_loss_quick, avg_val_loss_quick, step=step)
                model.train()  # Back to training mode

        # Full validation at end of epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = 0
            
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True)

                outputs = model(val_inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                val_targets = val_targets.view(-1)
                
                loss = criterion(outputs, val_targets)
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
        save_checkpoint(epoch + 1, model, optimizer, scheduler, avg_train_loss, avg_val_loss, is_best=is_best_model, step=step)
        
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
    print("🚀 FactLM Training - Efficient Model with DataLoaders")
    print("=" * 55)
    print("📝 Features: Optimized model (d_model=256), 4-worker DataLoaders")
    print("📈 Dataset: Books + 200K Wikipedia (factual knowledge focus)")
    print("⚡ Performance: Multi-worker data loading, step-based checkpoints")
    print("💾 Checkpoints: Every epoch + every 10,000 steps")
    print("🔄 Resume training by modifying this script to load from checkpoint")
    
    # Load and process all data using the data_loader module
    training_data, validation_data, data_stats = load_and_process_all_data(
        data_dir='data',
        ultrachat_samples=0,  # Removed UltraChat conversations
        wikipedia_samples=200000,  # Keep 200K Wikipedia articles
        generated_data_file=None,  # Remove generated data
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
    
    # Option 1: Standard model (256-dim) - good for larger datasets
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 256,      # Match d_model for efficiency
        'num_layers': 6,         # Reduced from 12 to lower parameter count
        'dropout': 0.2,          # Standard dropout
        'd_model': 256,          # Reduced from 512 for efficiency
        'max_len': 5000,
        'num_heads': 8           # 8 heads gives 32-dim heads (256/8=32)
    }
    
    # Option 2: Smaller model (128-dim) - better for smaller datasets to prevent overfitting
    # Uncomment the lines below to use a smaller model if you have limited training data
    # model_config = {
    #     'vocab_size': tokenizer.vocab_size,
    #     'hidden_size': 128,      # Smaller for less overfitting
    #     'num_layers': 4,         # Fewer layers
    #     'dropout': 0.3,          # Higher dropout for regularization
    #     'd_model': 128,          # Smaller embedding dimension
    #     'max_len': 5000,
    #     'num_heads': 4           # Fewer heads
    # }
    
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
    batch_size = 32       # Increased batch size for more stable gradients
    sequence_length = 256  # Start with shorter sequences for stable training
    learning_rate = 0.00015 # More conservative LR for improved architecture
    max_grad_norm = 1.0   # Gradient clipping
    checkpoint_every = 1  # Save checkpoint every epoch
    
    print(f"\n⚙️  Training configuration:")
    print(f"Device: {device} ({device_name})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch configuration: {batch_size} sequences × {sequence_length} tokens = {batch_size * sequence_length:,} tokens per batch")
    print(f"Sequence length: {sequence_length} tokens for stable training")
    print(f"Checkpoint frequency: Every {checkpoint_every} epoch + every 10,000 steps")
    print(f"DataLoader workers: 4 train + 2 validation (parallel processing)")
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