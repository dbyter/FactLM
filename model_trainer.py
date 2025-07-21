# Requirements: pip install torch transformers
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from factlm_model import FactLM
import re
import glob
import os
from datetime import datetime

def train_model(model, train_data, val_data, epochs, batch_size, learning_rate, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # Use NLLLoss since model outputs log_softmax

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_data) - batch_size, batch_size):
            inputs = train_data[i:i+batch_size].unsqueeze(0)  # Add batch dimension
            targets = train_data[i+1:i+batch_size+1]

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [1, seq_len, vocab_size]
            outputs = outputs.squeeze(0)  # Remove batch dimension: [seq_len, vocab_size]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = 0
            
            for i in range(0, len(val_data) - batch_size, batch_size):
                inputs = val_data[i:i+batch_size].unsqueeze(0)
                targets = val_data[i+1:i+batch_size+1]

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                outputs = outputs.squeeze(0)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model


def generate_text(model, start_string, max_length, temperature=0.5, tokenizer=None):
    """Generate text using the trained model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize the start string
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text generation")
    
    tokens = tokenizer.encode(start_string, add_special_tokens=True)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            # Get the last token's logits and apply temperature
            logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we hit an end token (you might want to define this)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated tokens
    generated_tokens = input_ids[0].cpu().tolist()
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
    """Tokenize book text and split into train/validation"""
    # Split text into chunks for better processing
    sentences = re.split(r'[.!?]+', book_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print(f"Processing {len(sentences)} sentences...")
    
    # Tokenize and concatenate
    all_tokens = []
    processed_sentences = 0
    
    for sentence in sentences:
        if len(sentence) > 10:  # Skip very short sentences
            tokens = tokenizer.encode(sentence + '.', add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)  # Add sentence separator
            processed_sentences += 1
    
    print(f"Processed {processed_sentences} sentences into {len(all_tokens)} tokens")
    
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
    utc_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_UTC')
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
        f.write(f"- Dropout: {model_config['dropout']}\n")
        f.write(f"- Max Length: {model_config['max_len']}\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"- Epochs: {training_stats['epochs']}\n")
        f.write(f"- Batch Size: {training_stats['batch_size']}\n")
        f.write(f"- Learning Rate: {training_stats['learning_rate']}\n")
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
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.2,
        'd_model': 1024,
        'max_len': 5000
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
    
    epochs = 20  # Reduced for larger dataset
    batch_size = 128  # Increased batch size for efficiency
    learning_rate = 0.0005  # Slightly lower for larger dataset
    
    print(f"Using device: {device} ({device_name})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    print("Starting training...")
    
    # Collect training statistics
    training_stats = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_tokens': len(training_data),
        'val_tokens': len(validation_data),
        'total_params': sum(p.numel() for p in model.parameters()),
        'book_files': book_files
    }
    
    trained_model = train_model(
        model=model,
        train_data=training_data,
        val_data=validation_data,
        epochs=epochs,
        batch_size=batch_size, 
        learning_rate=learning_rate,
        device=device
    )
    
    # Save the trained model
    print("\nSaving trained model...")
    model_path, metadata_path = save_model(
        model=trained_model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_stats=training_stats,
        device_name=device_name
    )
    
    # Generate some text
    print("\nGenerating text...")
    test_prompts = [
        "I am",
        "Where is",
        "What is",
        "Who is"
    ]
    
    for prompt in test_prompts:
        generated = generate_text(
            model=trained_model,
            start_string=prompt,
            max_length=50,
            temperature=0.8,
            tokenizer=tokenizer
        )
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {generated}")
        print("-" * 50)