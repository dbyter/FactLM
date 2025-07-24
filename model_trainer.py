# Requirements: pip install torch transformers datasets
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from datasets import load_dataset
from factlm_model import FactLM
import re
import glob
import os
import math
import random
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


def generate_text(model, start_string, max_length, temperature=0.8, tokenizer=None, repetition_penalty=1.2, top_k=50, top_p=0.9):
    """Generate text using the trained model with advanced sampling and anti-repetition"""
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
            
            # Advanced repetition penalty - stronger for recent tokens
            if repetition_penalty != 1.0:
                generated_list = generated_tokens[0].tolist()
                
                # Apply different penalties based on recency
                for i, token_id in enumerate(generated_list):
                    # Distance from current position (recent tokens get higher penalty)
                    distance = len(generated_list) - i
                    
                    if distance <= 5:  # Very recent tokens
                        penalty = repetition_penalty * 1.5
                    elif distance <= 15:  # Recent tokens
                        penalty = repetition_penalty * 1.2
                    else:  # Older tokens
                        penalty = repetition_penalty
                    
                    logits[token_id] /= penalty
            
            # N-gram repetition penalty - prevent repeated phrases
            if len(generated_tokens[0]) >= 6:
                generated_list = generated_tokens[0].tolist()
                
                # Check for 2-gram, 3-gram, and 4-gram repetitions
                for n in [2, 3, 4]:
                    if len(generated_list) >= n:
                        recent_ngram = tuple(generated_list[-n:])
                        
                        # Count occurrences of this n-gram in recent history
                        search_length = min(50, len(generated_list) - n)
                        count = 0
                        for i in range(len(generated_list) - n - search_length, len(generated_list) - n):
                            if i >= 0 and tuple(generated_list[i:i+n]) == recent_ngram:
                                count += 1
                        
                        # Apply penalty to tokens that would complete repeated n-grams
                        if count > 0:
                            # Look ahead: what tokens would complete this n-gram pattern?
                            for vocab_token_id in range(logits.size(0)):
                                # Check if adding this token would create a repeated n-gram
                                test_ngram = recent_ngram[1:] + (vocab_token_id,)
                                test_count = 0
                                for i in range(max(0, len(generated_list) - 50), len(generated_list) - n + 1):
                                    if tuple(generated_list[i:i+n]) == test_ngram:
                                        test_count += 1
                                
                                if test_count > 0:
                                    # Apply escalating penalty based on how many times we've seen this
                                    ngram_penalty = 1.5 + (0.5 * test_count)
                                    logits[vocab_token_id] /= ngram_penalty
            
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
            
            # Enhanced repetition detection
            if step > 8:
                generated_list = generated_tokens[0].tolist()
                
                # Check for immediate token repetition (same token multiple times)
                if len(generated_list) >= 3:
                    last_3 = generated_list[-3:]
                    if len(set(last_3)) == 1:  # Same token 3 times in a row
                        break
                
                # Check for alternating pattern (A-B-A-B...)
                if len(generated_list) >= 6:
                    last_6 = generated_list[-6:]
                    if (last_6[0] == last_6[2] == last_6[4] and 
                        last_6[1] == last_6[3] == last_6[5] and 
                        last_6[0] != last_6[1]):
                        break
                
                # Check for low diversity in recent tokens
                if step > 15:
                    recent_window = generated_list[-15:]
                    unique_ratio = len(set(recent_window)) / len(recent_window)
                    if unique_ratio < 0.4:  # Less than 40% unique tokens
                        break
                
                # Check for exact phrase repetition
                if len(generated_list) >= 8:
                    for phrase_len in [3, 4, 5]:
                        if len(generated_list) >= phrase_len * 2:
                            recent_phrase = generated_list[-phrase_len:]
                            prev_phrase = generated_list[-phrase_len*2:-phrase_len]
                            if recent_phrase == prev_phrase:
                                break
            
            # Add token to sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
            
            # Enhanced sentence ending detection
            if step > 5:
                # Get the actual token text to check for sentence endings
                current_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Check if we just completed a sentence
                if any(current_text.rstrip().endswith(punct) for punct in ['.', '!', '?', ':"', "'"]):
                    # Stop after sentence ending if we've generated enough
                    if step > 20:
                        break
                    # Or if the sentence seems complete and long enough
                    elif step > 10 and len(current_text.split()) >= 8:
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


def load_ultrachat_data(dataset_name="stingning/ultrachat", num_samples=50000, seed=42):
    """Load and sample UltraChat dataset from Hugging Face"""
    print(f"Loading UltraChat dataset: {dataset_name}")
    print(f"Sampling {num_samples:,} conversations (seed={seed})")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Use train split if available, otherwise use the first available split
        if 'train' in dataset:
            data_split = dataset['train']
        else:
            split_name = list(dataset.keys())[0]
            data_split = dataset[split_name]
            print(f"Using split: {split_name}")
        
        print(f"Total conversations in dataset: {len(data_split):,}")
        
        # Sample conversations if dataset is larger than requested
        if len(data_split) > num_samples:
            print(f"Sampling {num_samples:,} conversations from {len(data_split):,} total...")
            # Set seed for reproducible sampling
            random.seed(seed)
            indices = random.sample(range(len(data_split)), num_samples)
            sampled_data = data_split.select(indices)
        else:
            print(f"Using all {len(data_split):,} conversations")
            sampled_data = data_split
        
        return sampled_data
        
    except Exception as e:
        print(f"❌ Error loading UltraChat dataset: {e}")
        print("   Make sure you have the datasets library installed: pip install datasets")
        return None


def process_ultrachat_conversations(ultrachat_data, tokenizer):
    """Convert UltraChat conversations to training format"""
    if ultrachat_data is None:
        return []
    
    print(f"Processing {len(ultrachat_data):,} UltraChat conversations...")
    
    all_tokens = []
    processed_conversations = 0
    
    for example in ultrachat_data:
        try:
            # UltraChat format: each example has a 'data' field with conversation list
            conversation = example.get('data', [])
            
            if not conversation or len(conversation) < 2:
                continue
            
            # Format conversation into a readable text format
            conversation_text = ""
            
            for i, message in enumerate(conversation):
                if i == 0:
                    # First message is usually the human question
                    conversation_text += f"Human: {message}\n\n"
                elif i == 1:
                    # Second message is usually the assistant response
                    conversation_text += f"Assistant: {message}\n\n"
                elif i % 2 == 0:
                    # Even indices are human messages
                    conversation_text += f"Human: {message}\n\n"
                else:
                    # Odd indices are assistant messages
                    conversation_text += f"Assistant: {message}\n\n"
            
            # Add conversation end marker
            conversation_text += "<|endofconversation|>\n\n"
            
            # Tokenize the conversation
            tokens = tokenizer.encode(conversation_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            
            # Add EOS token between conversations
            all_tokens.append(tokenizer.eos_token_id)
            
            processed_conversations += 1
            
            # Progress reporting
            if processed_conversations % 5000 == 0:
                print(f"  Processed {processed_conversations:,} conversations...")
                
        except Exception as e:
            print(f"⚠️  Error processing conversation: {e}")
            continue
    
    print(f"✅ Processed {processed_conversations:,} UltraChat conversations into {len(all_tokens):,} tokens")
    
    return all_tokens


def combine_training_data(book_tokens, ultrachat_tokens, tokenizer, train_split=0.8):
    """Combine book and UltraChat data into training/validation sets"""
    print(f"\nCombining training data:")
    print(f"  Book tokens: {len(book_tokens):,}")
    print(f"  UltraChat tokens: {len(ultrachat_tokens):,}")
    
    # Combine all tokens
    all_tokens = book_tokens + ultrachat_tokens
    total_tokens = len(all_tokens)
    
    print(f"  Total combined tokens: {total_tokens:,}")
    
    # Shuffle the combined data for better training
    random.shuffle(all_tokens)
    print(f"  Data shuffled for better training distribution")
    
    # Convert to tensor
    full_data = torch.tensor(all_tokens, dtype=torch.long)
    
    # Split into train and validation
    split_idx = int(len(full_data) * train_split)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    print(f"  Train/validation split: {len(train_data):,} / {len(val_data):,} tokens ({train_split:.1%} / {1-train_split:.1%})")
    
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
            
            if 'ultrachat_tokens' in training_stats and training_stats['ultrachat_tokens'] > 0:
                f.write(f"\nUltraChat Data:\n")
                f.write(f"- Dataset: stingning/ultrachat\n")
                f.write(f"- Conversations: {training_stats.get('ultrachat_conversations', 0):,}\n")
                f.write(f"- Tokens: {training_stats['ultrachat_tokens']:,}\n")
                f.write(f"- Book tokens: {training_stats.get('book_tokens', 0):,}\n")
                f.write(f"- Total tokens ratio: {training_stats['ultrachat_tokens'] / (training_stats.get('book_tokens', 1) + training_stats['ultrachat_tokens']):.1%} UltraChat, {training_stats.get('book_tokens', 0) / (training_stats.get('book_tokens', 1) + training_stats['ultrachat_tokens']):.1%} Books\n")
    
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
    print("🚀 FactLM Training with Books + UltraChat Data")
    print("=" * 60)
    print("📝 Features: Multi-source training, automatic checkpointing every 5 epochs")
    print("💾 Checkpoints saved to: checkpoints/training_TIMESTAMP/")
    print("🔄 Resume training by modifying this script to load from checkpoint")
    
    # Load books
    print("\n📚 Loading and processing book data...")
    book_text, book_files = load_all_books('data')
    print(f"Sample book text: {book_text[:200]}...")
    
    # Create Hugging Face tokenizer (using GPT-2 tokenizer)
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}")
    
    # Process book data into tokens (but don't split yet)
    print("\n📖 Tokenizing book data...")
    book_tokens = []
    chunk_size = 10000  # characters per chunk
    
    for i in range(0, len(book_text), chunk_size):
        chunk = book_text[i:i + chunk_size]
        
        # Add overlap to avoid splitting words/sentences awkwardly
        if i > 0 and i + chunk_size < len(book_text):
            overlap_start = max(0, i - 200)
            overlap_text = book_text[overlap_start:i]
            
            # Find last sentence ending
            last_period = max(overlap_text.rfind('.'), overlap_text.rfind('!'), overlap_text.rfind('?'))
            if last_period > 0:
                chunk = book_text[overlap_start + last_period + 1:i + chunk_size]
        
        # Tokenize chunk
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        book_tokens.extend(tokens)
        
        # Add EOS token between chunks
        if i + chunk_size < len(book_text):
            book_tokens.append(tokenizer.eos_token_id)
    
    print(f"Book data: {len(book_tokens):,} tokens")
    
    # Load UltraChat data
    print("\n💬 Loading UltraChat dataset...")
    ultrachat_data = load_ultrachat_data(
        dataset_name="stingning/ultrachat",
        num_samples=25000,  # Adjust this number based on your needs and resources
        seed=42
    )
    
    # Process UltraChat conversations
    ultrachat_tokens = []
    if ultrachat_data is not None:
        print("\n🔄 Processing UltraChat conversations...")
        ultrachat_tokens = process_ultrachat_conversations(ultrachat_data, tokenizer)
    else:
        print("⚠️  Skipping UltraChat data due to loading error")
    
    # Combine all training data
    print("\n🔗 Combining training data...")
    training_data, validation_data = combine_training_data(
        book_tokens, ultrachat_tokens, tokenizer, train_split=0.8
    )
    
    print(f"\n📊 Final dataset statistics:")
    print(f"Training data shape: {training_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Sample tokens: {training_data[:20].tolist()}")
    print(f"Sample decoded: {tokenizer.decode(training_data[:50])}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    # Adjust model size based on larger combined dataset
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 256,      # Increased from 128
        'num_layers': 8,         # Increased from 6
        'dropout': 0.2,          # Reduced from 0.3 for larger dataset
        'd_model': 512,          # Increased from 256
        'max_len': 5000,
        'num_heads': 8           # 512 / 8 = 64 head dimension
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
    
    epochs = 30           # Reduced epochs since we have more data
    batch_size = 32       # Reduced batch size due to larger model
    sequence_length = 256  # Keep reasonable sequence length
    learning_rate = 0.0002 # Lower learning rate for stability
    max_grad_norm = 1.0   # Gradient clipping
    checkpoint_every = 5  # Save checkpoint every 5 epochs
    
    print(f"\n⚙️  Training configuration:")
    print(f"Device: {device} ({device_name})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch configuration: {batch_size} sequences × {sequence_length} tokens = {batch_size * sequence_length:,} tokens per batch")
    print(f"Checkpoint frequency: Every {checkpoint_every} epochs")
    
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
        'book_files': book_files,
        'book_tokens': len(book_tokens),
        'ultrachat_tokens': len(ultrachat_tokens),
        'ultrachat_conversations': len(ultrachat_data) if ultrachat_data else 0
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