# Requirements: pip install torch transformers
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from factlm_model import FactLM
import re
import glob
import os
import math
from datetime import datetime

def train_model(model, train_data, val_data, epochs, batch_size, sequence_length, learning_rate, device, max_grad_norm=1.0):
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
        sequence_length = min(sequence_length, 256)  # Cap sequence length
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
    
    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.1, step / warmup_steps)  # Minimum 10% of base LR
        else:
            # Cosine decay after warmup  
            if total_steps <= warmup_steps:
                return 1.0  # No decay if no steps after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(progress * math.pi))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.NLLLoss()  # Use NLLLoss since model outputs log_softmax

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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}, LR: {current_lr:.6f} ⭐ NEW BEST!")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}, LR: {current_lr:.6f} (patience: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break

    return model, batch_size, sequence_length


def generate_text(model, start_string, max_length, temperature=0.8, tokenizer=None, repetition_penalty=1.2, top_k=50, top_p=0.9):
    """Generate text using the trained model with advanced sampling"""
    model.eval()
    device = next(model.parameters()).device
    
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text generation")
    
    # Tokenize the start string
    tokens = tokenizer.encode(start_string, add_special_tokens=True)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_length):
            # Use only recent context to avoid memory issues
            context_length = min(generated_tokens.size(1), 256)
            context = generated_tokens[:, -context_length:]
            
            outputs = model(context)
            logits = outputs[0, -1, :].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens[0].tolist()):
                    logits[token_id] /= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Stop conditions
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Stop if we're generating too much repetition (simple check)
            if step > 10:
                recent_tokens = generated_tokens[0, -10:].tolist()
                if len(set(recent_tokens)) <= 2:  # Too much repetition
                    break
            
            # Add token to sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
            
            # Early stopping for natural sentence endings
            if step > 5 and next_token.item() in [tokenizer.encode('.')[0], tokenizer.encode('!')[0], tokenizer.encode('?')[0]]:
                # Stop after sentence ending if we've generated enough
                if step > 15:
                    break
    
    # Decode the generated tokens
    generated_tokens = generated_tokens[0].cpu().tolist()
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def load_and_clean_book(file_path):
    """Load and clean the Project Gutenberg book text"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lines to find content boundaries
    lines = content.split('\n')
    
    # Find where the actual book content starts (after "PREFACE" or similar)
    start_idx = 0
    for i, line in enumerate(lines):
        if 'PREFACE' in line.upper() or 'CHAPTER' in line.upper():
            start_idx = i
            break
    
    # Find where the book content ends (before Project Gutenberg footer)
    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if 'gutenberg' in line.lower() or 'copyright' in line.lower():
            end_idx = i
            break
    
    # Extract the main content
    book_lines = lines[start_idx:end_idx]
    text = '\n'.join(book_lines)
    
    # Clean the text
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newlines
    text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces/tabs to single space
    
    # Remove page numbers and chapter headers that are isolated
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*CHAPTER [IVXLC\d]+\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def load_all_books(data_dir='data'):
    """Load and combine all book*.txt files in the data directory"""
    # Find all book files
    book_pattern = os.path.join(data_dir, 'book*.txt')
    book_files = glob.glob(book_pattern)
    book_files.sort()  # Sort for consistent ordering
    
    if not book_files:
        raise FileNotFoundError(f"No book*.txt files found in {data_dir} directory")
    
    print(f"Found {len(book_files)} book files:")
    for file in book_files:
        print(f"  - {file}")
    
    # Load and combine all books
    combined_text = ""
    for book_file in book_files:
        print(f"Loading {book_file}...")
        book_text = load_and_clean_book(book_file)
        combined_text += book_text + "\n\n"  # Add separator between books
        print(f"  - {len(book_text)} characters loaded")
    
    combined_text = combined_text.strip()
    print(f"Total combined text: {len(combined_text)} characters")
    
    return combined_text, book_files


def prepare_book_data(book_text, tokenizer, train_split=0.8):
    """Tokenize book text and split into train/validation - improved to preserve more data"""
    print(f"Processing {len(book_text)} characters of text...")
    
    # Split into chunks of reasonable size for tokenization (avoid memory issues)
    chunk_size = 10000  # characters per chunk
    all_tokens = []
    
    for i in range(0, len(book_text), chunk_size):
        chunk = book_text[i:i + chunk_size]
        
        # Add overlap to avoid splitting words/sentences awkwardly
        if i > 0 and i + chunk_size < len(book_text):
            # Look back up to 200 chars for a good split point
            overlap_start = max(0, i - 200)
            overlap_text = book_text[overlap_start:i]
            
            # Find last sentence ending
            last_period = max(overlap_text.rfind('.'), overlap_text.rfind('!'), overlap_text.rfind('?'))
            if last_period > 0:
                chunk = book_text[overlap_start + last_period + 1:i + chunk_size]
        
        # Tokenize chunk directly - no sentence splitting to preserve more data
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        all_tokens.extend(tokens)
        
        # Add EOS token between chunks to maintain document structure
        if i + chunk_size < len(book_text):
            all_tokens.append(tokenizer.eos_token_id)
    
    print(f"Processed text into {len(all_tokens)} tokens (much better!)")
    
    # Convert to tensor
    full_data = torch.tensor(all_tokens, dtype=torch.long)
    
    # Split into train and validation
    split_idx = int(len(full_data) * train_split)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    print(f"Train/validation split: {len(train_data)} / {len(val_data)} tokens ({train_split:.1%} / {1-train_split:.1%})")
    
    return train_data, val_data


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
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    
    return model_path, metadata_path


def load_saved_model(model_path, tokenizer):
    """Load a previously saved model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    
    # Initialize model with saved configuration
    model = FactLM(**model_config)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"   - Trained on: {checkpoint['timestamp_utc']}")
    print(f"   - Device used: {checkpoint['device_used']}")
    print(f"   - Vocabulary size: {checkpoint['vocab_size']:,}")
    
    return model, checkpoint


# Example usage and data setup
if __name__ == "__main__":
    print("Loading and processing book data...")
    
    # Load all books
    book_text, book_files = load_all_books('data')
    print(f"Sample text: {book_text[:200]}...")
    
    # Create Hugging Face tokenizer (using GPT-2 tokenizer)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}")
    
    # Prepare training data
    print("Tokenizing book data...")
    training_data, validation_data = prepare_book_data(book_text, tokenizer)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Sample tokens: {training_data[:20].tolist()}")
    print(f"Sample decoded: {tokenizer.decode(training_data[:50])}")
    
    # Initialize model
    print("Initializing model...")
    # Much smaller model for the dataset size
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 128,
        'num_layers': 6,  # More layers, smaller width
        'dropout': 0.3,  # Higher dropout for regularization
        'd_model': 256,  # Much smaller
        'max_len': 5000,
        'num_heads': 8   # 256 / 8 = 32 head dimension
    }
    
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
    
    epochs = 50  # More epochs for better convergence
    batch_size = 64   # Moderate batch size for better generalization
    sequence_length = 256  # Reasonable sequence length
    learning_rate = 0.0003  # Lower, more conservative learning rate
    max_grad_norm = 1.0  # Tighter gradient clipping
    
    print(f"Using device: {device} ({device_name})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial batch configuration: {batch_size} sequences × {sequence_length} tokens = {batch_size * sequence_length:,} tokens per batch")
    
    # Check if configuration is reasonable for dataset size
    total_tokens_needed = batch_size * sequence_length
    if total_tokens_needed > len(training_data) // 2:
        print(f"⚠️  Batch size may be too large for dataset ({total_tokens_needed:,} tokens per batch vs {len(training_data):,} total tokens)")
        print("   The training function will auto-adjust if needed...")
    
    estimated_memory_mb = (batch_size * sequence_length * model_config['d_model'] * 4) // (1024**2)
    print(f"Estimated GPU memory usage: ~{estimated_memory_mb}MB per batch")
    
    if estimated_memory_mb > 8000:  # More than 8GB per batch
        print("⚠️  High memory usage detected! If you get OOM errors, reduce batch_size or sequence_length")
    
    # Train the model
    print("Starting training...")
    
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
        'book_files': book_files
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
        max_grad_norm=max_grad_norm
    )
    
    # Update training stats with final batch configuration
    training_stats['batch_size'] = final_batch_size
    training_stats['sequence_length'] = final_sequence_length
    
    # Save the trained model
    print("\nSaving trained model...")
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