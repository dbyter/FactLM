"""
FactLM Data Loading and Processing Module

This module handles loading and processing of training data from multiple sources:
- Project Gutenberg books (text files)
- UltraChat conversational data (Hugging Face datasets)
- Generated training data (from generate_training_data.py)
- Wikipedia articles (Hugging Face datasets)

Functions are designed to be used by model_trainer.py and other training scripts.
"""

import torch
import re
import glob
import os
import random
import json
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


def load_wikipedia_data(dataset_name="wikimedia/wikipedia", subset="20231101.en", num_samples=50000, seed=42):
    """Load and sample Wikipedia dataset from Hugging Face"""
    print(f"Loading Wikipedia dataset: {dataset_name}")
    print(f"Using subset: {subset}")
    print(f"Sampling {num_samples:,} articles (seed={seed})")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, subset)
        print(f"Wikipedia dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Use train split (Wikipedia only has train split)
        data_split = dataset['train']
        print(f"Total articles in dataset: {len(data_split):,}")
        
        # Sample articles if dataset is larger than requested
        if len(data_split) > num_samples:
            print(f"Sampling {num_samples:,} articles from {len(data_split):,} total...")
            # Set seed for reproducible sampling
            random.seed(seed)
            indices = random.sample(range(len(data_split)), num_samples)
            sampled_data = data_split.select(indices)
        else:
            print(f"Using all {len(data_split):,} articles")
            sampled_data = data_split
        
        return sampled_data
        
    except Exception as e:
        print(f"❌ Error loading Wikipedia dataset: {e}")
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


def load_generated_training_data(file_path="generated_training_data.json"):
    """Load generated training data from generate_training_data.py"""
    if not os.path.exists(file_path):
        print(f"⚠️  Generated training data file not found: {file_path}")
        return None
    
    try:
        print(f"📊 Loading generated training data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
        
        print(f"✅ Loaded {len(generated_data):,} generated conversations")
        
        # Validate data format
        valid_conversations = []
        for i, conversation in enumerate(generated_data):
            if 'data' in conversation and len(conversation['data']) == 2:
                valid_conversations.append(conversation)
            else:
                print(f"   ⚠️ Skipping malformed conversation {i}")
        
        print(f"✅ Validated {len(valid_conversations):,} conversations")
        return valid_conversations
        
    except Exception as e:
        print(f"❌ Error loading generated training data: {e}")
        return None


def process_generated_conversations(generated_data, tokenizer):
    """Convert generated training conversations to training format"""
    if generated_data is None:
        return []
    
    print(f"Processing {len(generated_data):,} generated conversations...")
    
    all_tokens = []
    processed_conversations = 0
    
    # Limit generated data to prevent overfitting
    max_generated_conversations = min(len(generated_data), 5000)  # Limit to 5K conversations
    if len(generated_data) > max_generated_conversations:
        print(f"   Limiting generated data to {max_generated_conversations:,} conversations to prevent overfitting")
        generated_data = generated_data[:max_generated_conversations]
    
    for example in generated_data:
        try:
            # Generated data format: each example has a 'data' field with [prompt, response]
            conversation = example.get('data', [])
            
            if not conversation or len(conversation) != 2:
                continue
            
            # Format conversation - use raw text without Human/Assistant prefixes
            prompt, response = conversation[0], conversation[1]
            
            # Check for potential issues in the data
            if len(prompt) < 10 or len(response) < 10:
                continue  # Skip very short conversations
            
            # Check for repetitive patterns in the response
            words = response.split()
            if len(words) > 3:
                # Check if response is just repeating the same word
                unique_words = set(words)
                if len(unique_words) / len(words) < 0.3:  # Too repetitive
                    continue
            
            conversation_text = f"{prompt}\n\n{response}\n\n"
            
            # Add conversation end marker
            conversation_text += "<|endofconversation|>\n\n"
            
            # Tokenize the conversation
            tokens = tokenizer.encode(conversation_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            
            # Add EOS token between conversations
            all_tokens.append(tokenizer.eos_token_id)
            
            processed_conversations += 1
            
            # Progress reporting
            if processed_conversations % 1000 == 0:
                print(f"  Processed {processed_conversations:,} conversations...")
                
        except Exception as e:
            print(f"⚠️  Error processing generated conversation: {e}")
            continue
    
    print(f"✅ Processed {processed_conversations:,} generated conversations into {len(all_tokens):,} tokens")
    
    return all_tokens


def process_wikipedia_articles(wikipedia_data, tokenizer):
    """Convert Wikipedia articles to training format"""
    if wikipedia_data is None:
        return []
    
    print(f"Processing {len(wikipedia_data):,} Wikipedia articles...")
    
    all_tokens = []
    processed_articles = 0
    
    for example in wikipedia_data:
        try:
            # Wikipedia format: each example has 'id', 'url', 'title', 'text' fields
            article_text = example.get('text', '')
            
            if not article_text or len(article_text.strip()) < 100:
                continue  # Skip very short articles
            
            # Use only the article text, no title or URL formatting
            formatted_text = f"{article_text}\n\n"
            
            # Add article end marker
            formatted_text += "<|endofarticle|>\n\n"
            
            # Tokenize the article
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            
            # Add EOS token between articles
            all_tokens.append(tokenizer.eos_token_id)
            
            processed_articles += 1
            
            # Progress reporting
            if processed_articles % 5000 == 0:
                print(f"  Processed {processed_articles:,} articles...")
                
        except Exception as e:
            print(f"⚠️  Error processing article: {e}")
            continue
    
    print(f"✅ Processed {processed_articles:,} Wikipedia articles into {len(all_tokens):,} tokens")
    
    return all_tokens


def combine_training_data(book_tokens, ultrachat_tokens, tokenizer, train_split=0.8):
    """Combine book and UltraChat data into training/validation sets"""
    print(f"\nCombining training data:")
    print(f"  Book tokens: {len(book_tokens):,}")
    print(f"  UltraChat tokens: {len(ultrachat_tokens):,}")
    
    # Combine all tokens - preserving sequence structure
    all_tokens = book_tokens + ultrachat_tokens
    total_tokens = len(all_tokens)
    
    print(f"  Total combined tokens: {total_tokens:,}")
    print(f"  ✅ Preserving text sequence structure (no token shuffling)")
    
    # Convert to tensor
    full_data = torch.tensor(all_tokens, dtype=torch.long)
    
    # Split into train and validation
    split_idx = int(len(full_data) * train_split)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    print(f"  Train/validation split: {len(train_data):,} / {len(val_data):,} tokens ({train_split:.1%} / {1-train_split:.1%})")
    
    return train_data, val_data


def combine_all_training_data(book_tokens, ultrachat_tokens, generated_tokens, wikipedia_tokens, tokenizer, train_split=0.8):
    """Combine book, UltraChat, generated, and Wikipedia data into training/validation sets"""
    print(f"\nCombining all training data:")
    print(f"  Book tokens: {len(book_tokens):,}")
    print(f"  UltraChat tokens: {len(ultrachat_tokens):,}")
    print(f"  Generated tokens: {len(generated_tokens):,}")
    print(f"  Wikipedia tokens: {len(wikipedia_tokens):,}")
    
    # Combine all tokens - preserving sequence structure
    all_tokens = book_tokens + ultrachat_tokens + generated_tokens + wikipedia_tokens
    total_tokens = len(all_tokens)
    
    print(f"  Total combined tokens: {total_tokens:,}")
    print(f"  ✅ Preserving text sequence structure (no token shuffling)")
    
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


def print_data_samples(data_type, samples, tokenizer, max_samples=10):
    """Print samples from different data sources for inspection"""
    print(f"\n📋 Sample {data_type} data:")
    print("-" * 60)
    
    for i, sample in enumerate(samples[:max_samples]):
        print(f"\n{i+1:2d}. {sample}")
        if len(sample) > 200:  # Truncate very long samples
            print(f"    [... truncated after 200 chars]")
    
    if len(samples) > max_samples:
        print(f"\n   ... and {len(samples) - max_samples} more samples")


def print_conversation_samples(data_type, conversations, max_samples=10):
    """Print conversation samples showing input/output pairs"""
    print(f"\n📋 Sample {data_type} conversations:")
    print("-" * 60)
    
    count = 0
    for i, conversation in enumerate(conversations):
        if count >= max_samples:
            break
            
        try:
            conv_data = conversation.get('data', [])
            
            if data_type == "UltraChat":
                # UltraChat format: 'data' contains list of messages alternating human/assistant
                if len(conv_data) >= 2:
                    prompt = conv_data[0]  # First message (human)
                    response = conv_data[1]  # Second message (assistant)
                    
                    # Truncate long text for readability
                    prompt_display = prompt[:150] + "..." if len(prompt) > 150 else prompt
                    response_display = response[:200] + "..." if len(response) > 200 else response
                    
                    print(f"\n{count+1:2d}. HUMAN:     {prompt_display}")
                    print(f"   ASSISTANT: {response_display}")
                    count += 1
                    
            else:
                # Generated data format: 'data' contains [prompt, response] pair
                if len(conv_data) >= 2:
                    prompt = conv_data[0]
                    response = conv_data[1]
                    
                    # Truncate long text for readability
                    prompt_display = prompt[:150] + "..." if len(prompt) > 150 else prompt
                    response_display = response[:200] + "..." if len(response) > 200 else response
                    
                    print(f"\n{count+1:2d}. INPUT:  {prompt_display}")
                    print(f"   OUTPUT: {response_display}")
                    count += 1
                    
        except Exception as e:
            continue
    
    if len(conversations) > max_samples:
        print(f"\n   ... and {len(conversations) - max_samples} more conversations")


def print_book_samples(book_text, tokenizer, max_samples=10):
    """Print samples from book text"""
    print(f"\n📋 Sample book text chunks:")
    print("-" * 60)
    
    # Split book text into chunks for sampling
    chunk_size = 300  # Characters per chunk
    chunks = []
    
    for i in range(0, min(len(book_text), chunk_size * max_samples * 3), chunk_size):
        chunk = book_text[i:i + chunk_size].strip()
        if len(chunk) > 50:  # Skip very short chunks
            chunks.append(chunk)
    
    for i, chunk in enumerate(chunks[:max_samples]):
        # Clean up the chunk for display
        clean_chunk = ' '.join(chunk.split())  # Remove extra whitespace
        if len(clean_chunk) > 250:
            clean_chunk = clean_chunk[:250] + "..."
        
        print(f"\n{i+1:2d}. {clean_chunk}")
    
    print(f"\n   Total book text: {len(book_text):,} characters")


def print_wikipedia_samples(wikipedia_data, max_samples=10):
    """Print samples from Wikipedia articles"""
    print(f"\n📋 Sample Wikipedia articles:")
    print("-" * 60)
    
    count = 0
    for i, article in enumerate(wikipedia_data):
        if count >= max_samples:
            break
            
        try:
            text = article.get('text', '')
            
            if len(text) > 50:  # Skip very short articles
                # Truncate text for readability
                text_display = text[:300] + "..." if len(text) > 300 else text
                # Clean up text formatting
                clean_text = ' '.join(text_display.split())
                
                print(f"\n{count+1:2d}. {clean_text}")
                count += 1
                    
        except Exception as e:
            continue
    
    if len(wikipedia_data) > max_samples:
        print(f"\n   ... and {len(wikipedia_data) - max_samples} more articles")


def load_and_process_all_data(data_dir='data', 
                             ultrachat_samples=0,  # Updated default to 0 (disabled)
                             wikipedia_samples=300000,  # Updated default to 300K
                             generated_data_file=None,  # Updated default to None (disabled)
                             train_split=0.8, 
                             seed=42):
    """
    Complete data loading pipeline - loads books, UltraChat, generated data, and Wikipedia, processes and combines them
    
    Args:
        data_dir (str): Directory containing book*.txt files
        ultrachat_samples (int): Number of UltraChat conversations to sample (default: 0 - disabled)
        wikipedia_samples (int): Number of Wikipedia articles to sample (default: 300K)
        generated_data_file (str): Path to generated training data JSON file (default: None - disabled)
        train_split (float): Fraction of data to use for training (rest for validation)
        seed (int): Random seed for reproducible sampling
    
    Returns:
        tuple: (train_data, val_data, data_stats) where data_stats contains metadata
    """
    from transformers import AutoTokenizer
    
    print("🔄 Starting data loading pipeline (BOOKS + WIKIPEDIA)...")
    print("   📚 UltraChat and Generated data disabled for focused factual training")
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load books
    print("\n📚 Loading book data...")
    book_text, book_files = load_all_books(data_dir)
    
    # Print book samples
    print_book_samples(book_text, tokenizer, max_samples=10)
    
    # Process book data into tokens
    print("\n📖 Tokenizing book data...")
    book_tokens = tokenize_text_data(book_text, tokenizer)
    
    # Load UltraChat data
    ultrachat_tokens = []
    if ultrachat_samples > 0:
        print("\n💬 Loading UltraChat dataset...")
        ultrachat_data = load_ultrachat_data(
            dataset_name="stingning/ultrachat",
            num_samples=ultrachat_samples,  # Use the parameter value
            seed=seed
        )
        
        # Print UltraChat samples (if loaded successfully)
        if ultrachat_data is not None:
            print_conversation_samples("UltraChat", ultrachat_data, max_samples=10)
        
        # Process UltraChat conversations
        if ultrachat_data is not None:
            print("\n🔄 Processing UltraChat conversations...")
            ultrachat_tokens = process_ultrachat_conversations(ultrachat_data, tokenizer)
        else:
            print("⚠️  Skipping UltraChat data due to loading error")
    else:
        print("\n💬 Skipping UltraChat data (ultrachat_samples=0)")
        ultrachat_data = None
    
    # Load and process generated training data
    generated_tokens = []
    if generated_data_file is not None:
        print(f"\n🤖 Loading generated training data from {generated_data_file}...")
        generated_data = load_generated_training_data(generated_data_file)
        
        # Print generated conversation samples (if loaded successfully)
        if generated_data is not None:
            print_conversation_samples("Generated", generated_data, max_samples=10)
        
        if generated_data is not None:
            print("\n🔄 Processing generated conversations...")
            generated_tokens = process_generated_conversations(generated_data, tokenizer)
        else:
            print("⚠️  No generated training data found - continuing without it")
    else:
        print("\n🤖 Skipping generated data (generated_data_file=None)")
        generated_data = None
    
    # Load Wikipedia data
    print("\n📚 Loading Wikipedia dataset...")
    wikipedia_data = load_wikipedia_data(
        dataset_name="wikimedia/wikipedia",
        subset="20231101.en",
        num_samples=wikipedia_samples, # Use the parameter value (now 300K)
        seed=seed
    )
    
    # Print Wikipedia samples (if loaded successfully)
    if wikipedia_data is not None:
        print_wikipedia_samples(wikipedia_data, max_samples=10)
    
    # Process Wikipedia articles
    wikipedia_tokens = []
    if wikipedia_data is not None:
        print("\n🔄 Processing Wikipedia articles...")
        wikipedia_tokens = process_wikipedia_articles(wikipedia_data, tokenizer)
    else:
        print("⚠️  Skipping Wikipedia data due to loading error")
    
    # Combine all training data (books + Wikipedia only)
    print("\n🔗 Combining training data (BOOKS + WIKIPEDIA)...")
    train_data, val_data = combine_all_training_data(
        book_tokens, ultrachat_tokens, generated_tokens, wikipedia_tokens, tokenizer, train_split=train_split
    )
    
    # Prepare statistics
    data_stats = {
        'tokenizer': tokenizer,
        'book_files': book_files,
        'book_tokens': len(book_tokens),
        'ultrachat_tokens': len(ultrachat_tokens),
        'ultrachat_conversations': len(ultrachat_data) if ultrachat_data else 0,
        'generated_tokens': len(generated_tokens),
        'generated_conversations': len(generated_data) if generated_data else 0,
        'wikipedia_tokens': len(wikipedia_tokens),
        'wikipedia_articles': len(wikipedia_data) if wikipedia_data else 0,
        'total_tokens': len(train_data) + len(val_data),
        'train_tokens': len(train_data),
        'val_tokens': len(val_data),
        'train_split': train_split
    }
    
    print(f"\n✅ Data loading complete!")
    print(f"   📊 Total tokens: {data_stats['total_tokens']:,}")
    print(f"   📚 Book tokens: {data_stats['book_tokens']:,} (ACTIVE)")
    if data_stats['ultrachat_tokens'] > 0:
        print(f"   💬 UltraChat tokens: {data_stats['ultrachat_tokens']:,} (ACTIVE)")
    else:
        print(f"   💬 UltraChat tokens: 0 (SKIPPED)")
    if data_stats['generated_tokens'] > 0:
        print(f"   🤖 Generated tokens: {data_stats['generated_tokens']:,} (ACTIVE)")
    else:
        print(f"   🤖 Generated tokens: 0 (SKIPPED)")
    print(f"   📚 Wikipedia tokens: {data_stats['wikipedia_tokens']:,} (ACTIVE)")
    print(f"   🔄 Train/Val split: {len(train_data):,} / {len(val_data):,}")
    
    # Show data source proportions
    if data_stats['total_tokens'] > 0:
        book_pct = (data_stats['book_tokens'] / data_stats['total_tokens']) * 100
        ultrachat_pct = (data_stats['ultrachat_tokens'] / data_stats['total_tokens']) * 100
        gen_pct = (data_stats['generated_tokens'] / data_stats['total_tokens']) * 100
        wiki_pct = (data_stats['wikipedia_tokens'] / data_stats['total_tokens']) * 100
        print(f"   📊 Data mix: {book_pct:.1f}% books, {ultrachat_pct:.1f}% UltraChat, {gen_pct:.1f}% generated, {wiki_pct:.1f}% Wikipedia")
    
    if data_stats['generated_tokens'] == 0 and generated_data_file is not None:
        print(f"\n⚠️  WARNING: No generated tokens found!")
        print(f"   Make sure to run generate_training_data.py first")
        print(f"   Looking for file: {generated_data_file}")
    
    return train_data, val_data, data_stats 