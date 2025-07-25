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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


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


def process_single_conversation(conversation_data):
    """Process a single UltraChat conversation - helper function for multithreading"""
    try:
        # UltraChat format: each example has a 'data' field with conversation list
        conversation = conversation_data.get('data', [])
        
        if not conversation or len(conversation) < 2:
            return None
        
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
        
        return conversation_text
        
    except Exception as e:
        return None


def process_ultrachat_conversations(ultrachat_data, tokenizer, max_workers=4):
    """Convert UltraChat conversations to training format using multithreading"""
    if ultrachat_data is None:
        return []
    
    print(f"Processing {len(ultrachat_data):,} UltraChat conversations using {max_workers} threads...")
    
    all_tokens = []
    processed_conversations = 0
    progress_lock = Lock()
    
    # Convert dataset to list for easier processing
    conversations_list = list(ultrachat_data)
    
    # Process conversations in parallel
    formatted_conversations = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversations for processing
        future_to_conversation = {
            executor.submit(process_single_conversation, conversation): i 
            for i, conversation in enumerate(conversations_list)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_conversation):
            try:
                formatted_conversation = future.result()
                if formatted_conversation is not None:
                    formatted_conversations.append(formatted_conversation)
                
                with progress_lock:
                    processed_conversations += 1
                    if processed_conversations % 5000 == 0:
                        print(f"  Processed {processed_conversations:,} conversations...")
                        
            except Exception as e:
                print(f"⚠️  Error processing conversation: {e}")
                continue
    
    print(f"✅ Formatted {len(formatted_conversations):,} conversations, now tokenizing...")
    
    # Tokenize all formatted conversations
    tokenization_batch_size = 1000
    
    for i in range(0, len(formatted_conversations), tokenization_batch_size):
        batch = formatted_conversations[i:i + tokenization_batch_size]
        
        for conversation_text in batch:
            try:
                # Tokenize the conversation
                tokens = tokenizer.encode(conversation_text, add_special_tokens=False)
                all_tokens.extend(tokens)
                
                # Add EOS token between conversations
                all_tokens.append(tokenizer.eos_token_id)
                
            except Exception as e:
                print(f"⚠️  Error tokenizing conversation: {e}")
                continue
        
        # Progress update for tokenization
        tokenized_so_far = min(i + tokenization_batch_size, len(formatted_conversations))
        if tokenized_so_far % 5000 == 0:
            print(f"  Tokenized {tokenized_so_far:,} conversations...")
    
    print(f"✅ Processed {len(formatted_conversations):,} UltraChat conversations into {len(all_tokens):,} tokens")
    
    return all_tokens


def process_ultrachat_conversations_legacy(ultrachat_data, tokenizer):
    """Convert UltraChat conversations to training format (legacy single-threaded version)"""
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


def process_single_generated_conversation(conversation_data):
    """Process a single generated conversation - helper function for multithreading"""
    try:
        # Generated data format: each example has a 'data' field with [prompt, response]
        conversation = conversation_data.get('data', [])
        
        if not conversation or len(conversation) != 2:
            return None
        
        # Format conversation - use raw text without Human/Assistant prefixes
        prompt, response = conversation[0], conversation[1]
        
        # Check for potential issues in the data
        if len(prompt) < 10 or len(response) < 10:
            return None  # Skip very short conversations
        
        # Check for repetitive patterns in the response
        words = response.split()
        if len(words) > 3:
            # Check if response is just repeating the same word
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Too repetitive
                return None
        
        conversation_text = f"{prompt}\n\n{response}\n\n"
        
        # Add conversation end marker
        conversation_text += "<|endofconversation|>\n\n"
        
        return conversation_text
        
    except Exception as e:
        return None


def process_generated_conversations(generated_data, tokenizer, max_workers=4):
    """Convert generated training conversations to training format using multithreading"""
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
    
    print(f"Processing {len(generated_data):,} generated conversations using {max_workers} threads...")
    
    progress_lock = Lock()
    
    # Process conversations in parallel
    formatted_conversations = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversations for processing
        future_to_conversation = {
            executor.submit(process_single_generated_conversation, conversation): i 
            for i, conversation in enumerate(generated_data)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_conversation):
            try:
                formatted_conversation = future.result()
                if formatted_conversation is not None:
                    formatted_conversations.append(formatted_conversation)
                
                with progress_lock:
                    processed_conversations += 1
                    if processed_conversations % 1000 == 0:
                        print(f"  Processed {processed_conversations:,} conversations...")
                        
            except Exception as e:
                print(f"⚠️  Error processing generated conversation: {e}")
                continue
    
    print(f"✅ Formatted {len(formatted_conversations):,} generated conversations, now tokenizing...")
    
    # Tokenize all formatted conversations
    tokenization_batch_size = 500  # Smaller batches for generated data
    
    for i in range(0, len(formatted_conversations), tokenization_batch_size):
        batch = formatted_conversations[i:i + tokenization_batch_size]
        
        for conversation_text in batch:
            try:
                # Tokenize the conversation
                tokens = tokenizer.encode(conversation_text, add_special_tokens=False)
                all_tokens.extend(tokens)
                
                # Add EOS token between conversations
                all_tokens.append(tokenizer.eos_token_id)
                
            except Exception as e:
                print(f"⚠️  Error tokenizing conversation: {e}")
                continue
        
        # Progress update for tokenization
        tokenized_so_far = min(i + tokenization_batch_size, len(formatted_conversations))
        if tokenized_so_far % 1000 == 0:
            print(f"  Tokenized {tokenized_so_far:,} conversations...")
    
    print(f"✅ Processed {len(formatted_conversations):,} generated conversations into {len(all_tokens):,} tokens")
    
    return all_tokens


def process_generated_conversations_legacy(generated_data, tokenizer):
    """Convert generated training conversations to training format (legacy single-threaded version)"""
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


def process_single_article(article_data):
    """Process a single Wikipedia article - helper function for multithreading"""
    try:
        article_text = article_data.get('text', '')
        
        if not article_text or len(article_text.strip()) < 100:
            return None  # Skip very short articles
        
        # Use only the article text, no title or URL formatting
        formatted_text = f"{article_text}\n\n"
        
        # Add article end marker
        formatted_text += "<|endofarticle|>\n\n"
        
        return formatted_text
        
    except Exception as e:
        return None


def process_wikipedia_articles(wikipedia_data, tokenizer, max_workers=4):
    """Convert Wikipedia articles to training format using multithreading"""
    if wikipedia_data is None:
        return []
    
    print(f"Processing {len(wikipedia_data):,} Wikipedia articles using {max_workers} threads...")
    
    all_tokens = []
    processed_articles = 0
    progress_lock = Lock()
    
    # Convert dataset to list for easier batching
    articles_list = list(wikipedia_data)
    
    # Process articles in parallel
    batch_size = max(1, len(articles_list) // (max_workers * 4))  # Create more batches than workers
    formatted_texts = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all articles for processing
        future_to_article = {
            executor.submit(process_single_article, article): i 
            for i, article in enumerate(articles_list)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_article):
            try:
                formatted_text = future.result()
                if formatted_text is not None:
                    formatted_texts.append(formatted_text)
                
                with progress_lock:
                    processed_articles += 1
                    # More frequent progress updates for large datasets
                    progress_interval = 10000 if len(articles_list) > 500000 else 5000
                    if processed_articles % progress_interval == 0:
                        percentage = (processed_articles / len(articles_list)) * 100
                        print(f"  Processed {processed_articles:,} articles... ({percentage:.1f}% complete)")
                        
            except Exception as e:
                print(f"⚠️  Error processing article: {e}")
                continue
    
    print(f"✅ Formatted {len(formatted_texts):,} articles, now tokenizing...")
    
    # Tokenize all formatted texts
    # We'll do this in batches to manage memory and show progress
    tokenization_batch_size = 1000
    
    for i in range(0, len(formatted_texts), tokenization_batch_size):
        batch = formatted_texts[i:i + tokenization_batch_size]
        
        for formatted_text in batch:
            try:
                # Tokenize the article
                tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                all_tokens.extend(tokens)
                
                # Add EOS token between articles
                all_tokens.append(tokenizer.eos_token_id)
                
            except Exception as e:
                print(f"⚠️  Error tokenizing article: {e}")
                continue
        
        # Progress update for tokenization
        tokenized_so_far = min(i + tokenization_batch_size, len(formatted_texts))
        progress_interval = 10000 if len(formatted_texts) > 500000 else 5000
        if tokenized_so_far % progress_interval == 0:
            percentage = (tokenized_so_far / len(formatted_texts)) * 100
            print(f"  Tokenized {tokenized_so_far:,} articles... ({percentage:.1f}% complete)")
    
    print(f"✅ Processed {len(formatted_texts):,} Wikipedia articles into {len(all_tokens):,} tokens")
    
    return all_tokens


def process_wikipedia_articles_chunked(wikipedia_data, tokenizer, max_workers=4, chunk_size=1000):
    """
    Alternative multithreaded approach that processes articles in chunks
    This can be more memory efficient for very large datasets
    """
    if wikipedia_data is None:
        return []
    
    print(f"Processing {len(wikipedia_data):,} Wikipedia articles in chunks of {chunk_size} using {max_workers} threads...")
    
    all_tokens = []
    total_processed = 0
    
    # Convert to list and split into chunks
    articles_list = list(wikipedia_data)
    
    def process_article_chunk(chunk):
        """Process a chunk of articles"""
        chunk_tokens = []
        chunk_processed = 0
        
        for article in chunk:
            try:
                article_text = article.get('text', '')
                
                if not article_text or len(article_text.strip()) < 100:
                    continue
                
                # Format the article
                formatted_text = f"{article_text}\n\n<|endofarticle|>\n\n"
                
                # Tokenize the article
                tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                chunk_tokens.extend(tokens)
                
                # Add EOS token between articles
                chunk_tokens.append(tokenizer.eos_token_id)
                
                chunk_processed += 1
                
            except Exception as e:
                continue
        
        return chunk_tokens, chunk_processed
    
    # Process chunks in parallel
    for i in range(0, len(articles_list), chunk_size):
        chunk = articles_list[i:i + chunk_size]
        chunks = [chunk[j:j + len(chunk)//max_workers + 1] for j in range(0, len(chunk), len(chunk)//max_workers + 1)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_article_chunk, chunk) for chunk in chunks if chunk]
            
            for future in as_completed(futures):
                try:
                    chunk_tokens, chunk_processed = future.result()
                    all_tokens.extend(chunk_tokens)
                    total_processed += chunk_processed
                except Exception as e:
                    print(f"⚠️  Error processing chunk: {e}")
        
        print(f"  Processed {total_processed:,} articles so far...")
    
    print(f"✅ Processed {total_processed:,} Wikipedia articles into {len(all_tokens):,} tokens")
    
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
                             wikipedia_samples=1000000,  # Updated default to 1M
                             generated_data_file=None,  # Updated default to None (disabled)
                             train_split=0.8, 
                             seed=42,
                             max_workers=4):  # Number of threads for Wikipedia processing
    """
    Complete data loading pipeline - loads books, UltraChat, generated data, and Wikipedia, processes and combines them
    
    Args:
        data_dir (str): Directory containing book*.txt files
        ultrachat_samples (int): Number of UltraChat conversations to sample (default: 0 - disabled)
        wikipedia_samples (int): Number of Wikipedia articles to sample (default: 1M)
        generated_data_file (str): Path to generated training data JSON file (default: None - disabled)
        train_split (float): Fraction of data to use for training (rest for validation)
        seed (int): Random seed for reproducible sampling
        max_workers (int): Number of threads for Wikipedia processing (default: 4, increase for faster processing)
    
    Returns:
        tuple: (train_data, val_data, data_stats) where data_stats contains metadata
    
    Note:
        Processing 1M Wikipedia articles will take significant time and memory.
        Consider using max_workers=8 or higher for faster processing on multi-core systems.
    """
    from transformers import AutoTokenizer
    
    print("🔄 Starting data loading pipeline (BOOKS + WIKIPEDIA)...")
    print("   📚 UltraChat and Generated data disabled for focused factual training")
    print(f"   🚀 Using {max_workers} threads for parallel processing (adjust max_workers parameter to tune performance)")
    print(f"   📊 Processing {wikipedia_samples:,} Wikipedia articles (this may take 10-30 minutes depending on your system)")
    
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
            print(f"\n🔄 Processing UltraChat conversations with {max_workers} threads...")
            ultrachat_tokens = process_ultrachat_conversations(ultrachat_data, tokenizer, max_workers=max_workers)
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
            print(f"\n🔄 Processing generated conversations with {max_workers} threads...")
            generated_tokens = process_generated_conversations(generated_data, tokenizer, max_workers=max_workers)
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
        num_samples=wikipedia_samples, # Use the parameter value (now 1M)
        seed=seed
    )
    
    # Print Wikipedia samples (if loaded successfully)
    if wikipedia_data is not None:
        print_wikipedia_samples(wikipedia_data, max_samples=10)
    
    # Process Wikipedia articles
    wikipedia_tokens = []
    if wikipedia_data is not None:
        print(f"\n🔄 Processing Wikipedia articles with {max_workers} threads...")
        wikipedia_tokens = process_wikipedia_articles(wikipedia_data, tokenizer, max_workers=max_workers)
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


def estimate_processing_time(wikipedia_samples=1000000, max_workers=4):
    """
    Estimate processing time for Wikipedia articles based on sample size and worker count
    
    Args:
        wikipedia_samples (int): Number of Wikipedia articles to process
        max_workers (int): Number of worker threads
    
    Returns:
        dict: Estimated times for different processing phases
    """
    import os
    
    # Base processing rates (articles per second) - these are rough estimates
    # Based on typical performance on modern systems
    base_download_rate = 2000  # articles/second for downloading
    base_processing_rate = 300  # articles/second for single-threaded processing
    base_tokenization_rate = 500  # articles/second for tokenization
    
    # Adjust for worker count (with diminishing returns)
    worker_efficiency = min(max_workers, os.cpu_count() or 4)
    if worker_efficiency > 4:
        # Efficiency drops off after 4 workers due to I/O bottlenecks
        worker_efficiency = 4 + (worker_efficiency - 4) * 0.5
    
    processing_rate = base_processing_rate * worker_efficiency
    tokenization_rate = base_tokenization_rate * worker_efficiency
    
    # Calculate estimated times
    download_time = wikipedia_samples / base_download_rate
    processing_time = wikipedia_samples / processing_rate  
    tokenization_time = wikipedia_samples / tokenization_rate
    total_time = download_time + processing_time + tokenization_time
    
    return {
        'samples': wikipedia_samples,
        'workers': max_workers,
        'effective_workers': worker_efficiency,
        'download_minutes': download_time / 60,
        'processing_minutes': processing_time / 60,
        'tokenization_minutes': tokenization_time / 60,
        'total_minutes': total_time / 60,
        'total_hours': total_time / 3600
    }


def print_processing_estimates():
    """Print processing time estimates for common configurations"""
    print("⏱️  Wikipedia Processing Time Estimates")
    print("=" * 60)
    
    configs = [
        (100000, 4, "100K articles, 4 workers (quick test)"),
        (300000, 4, "300K articles, 4 workers (medium dataset)"),  
        (1000000, 4, "1M articles, 4 workers (default)"),
        (1000000, 8, "1M articles, 8 workers (fast)"),
        (1000000, 12, "1M articles, 12 workers (very fast)"),
    ]
    
    print(f"{'Configuration':<35} {'Download':<10} {'Process':<10} {'Tokenize':<10} {'Total':<10}")
    print("-" * 75)
    
    for samples, workers, description in configs:
        est = estimate_processing_time(samples, workers)
        
        download_str = f"{est['download_minutes']:.1f}m"
        process_str = f"{est['processing_minutes']:.1f}m" 
        tokenize_str = f"{est['tokenization_minutes']:.1f}m"
        
        if est['total_hours'] >= 1:
            total_str = f"{est['total_hours']:.1f}h"
        else:
            total_str = f"{est['total_minutes']:.1f}m"
            
        print(f"{description:<35} {download_str:<10} {process_str:<10} {tokenize_str:<10} {total_str:<10}")
    
    print(f"\nNote: Times are rough estimates and will vary based on:")
    print(f"  - Internet connection speed (for download)")
    print(f"  - CPU performance and memory")
    print(f"  - System load and available resources")
    print(f"  - Article length and complexity")