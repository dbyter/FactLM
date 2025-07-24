"""
FactLM Data Loading and Processing Module

This module handles loading and processing of training data from multiple sources:
- Project Gutenberg books (text files)
- UltraChat conversational data (Hugging Face datasets)

Functions are designed to be used by model_trainer.py and other training scripts.
"""

import torch
import re
import glob
import os
import random
from datasets import load_dataset


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


def tokenize_text_data(text_data, tokenizer):
    """Tokenize raw text data using the provided tokenizer
    
    Args:
        text_data (str): Raw text to tokenize
        tokenizer: Tokenizer instance (e.g., from transformers)
    
    Returns:
        list: List of token IDs
    """
    # For small to medium texts, tokenize all at once for best quality
    if len(text_data) <= 500000:  # ~500K characters or less
        return tokenizer.encode(text_data, add_special_tokens=False)
    
    # For very large texts, split at sentence boundaries to preserve word integrity
    print(f"   Large text detected ({len(text_data):,} chars), using sentence-boundary chunking...")
    
    # Split into sentences first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text_data)
    
    all_tokens = []
    current_chunk = ""
    chunk_size_limit = 50000  # Character limit per chunk (larger than before)
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, process current chunk
        if len(current_chunk) + len(sentence) > chunk_size_limit and current_chunk:
            tokens = tokenizer.encode(current_chunk.strip(), add_special_tokens=False)
            all_tokens.extend(tokens)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Process the final chunk
    if current_chunk.strip():
        tokens = tokenizer.encode(current_chunk.strip(), add_special_tokens=False)
        all_tokens.extend(tokens)
    
    return all_tokens


def load_and_process_all_data(data_dir='data', ultrachat_samples=50000, train_split=0.8, seed=42):
    """
    Complete data loading pipeline - loads books and UltraChat data, processes and combines them
    
    Args:
        data_dir (str): Directory containing book*.txt files
        ultrachat_samples (int): Number of UltraChat conversations to sample
        train_split (float): Fraction of data to use for training (rest for validation)
        seed (int): Random seed for reproducible sampling
    
    Returns:
        tuple: (train_data, val_data, data_stats) where data_stats contains metadata
    """
    from transformers import AutoTokenizer
    
    print("🔄 Starting complete data loading pipeline...")
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load books
    print("\n📚 Loading book data...")
    book_text, book_files = load_all_books(data_dir)
    
    # Process book data into tokens
    print("\n📖 Tokenizing book data...")
    book_tokens = tokenize_text_data(book_text, tokenizer)
    
    # Load UltraChat data
    print("\n💬 Loading UltraChat dataset...")
    ultrachat_data = load_ultrachat_data(
        dataset_name="stingning/ultrachat",
        num_samples=ultrachat_samples,
        seed=seed
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
    train_data, val_data = combine_training_data(
        book_tokens, ultrachat_tokens, tokenizer, train_split=train_split
    )
    
    # Prepare statistics
    data_stats = {
        'tokenizer': tokenizer,
        'book_files': book_files,
        'book_tokens': len(book_tokens),
        'ultrachat_tokens': len(ultrachat_tokens),
        'ultrachat_conversations': len(ultrachat_data) if ultrachat_data else 0,
        'total_tokens': len(train_data) + len(val_data),
        'train_tokens': len(train_data),
        'val_tokens': len(val_data),
        'train_split': train_split
    }
    
    print(f"\n✅ Data loading complete!")
    print(f"   📊 Total tokens: {data_stats['total_tokens']:,}")
    print(f"   📚 Book tokens: {data_stats['book_tokens']:,}")
    print(f"   💬 UltraChat tokens: {data_stats['ultrachat_tokens']:,}")
    print(f"   🔄 Train/Val split: {len(train_data):,} / {len(val_data):,}")
    
    return train_data, val_data, data_stats 