#!/usr/bin/env python3
"""
Data Preparation Debug Script

This script tests the data preparation to identify any issues that could cause
the model to learn to repeat the last token instead of predicting the next token.
"""

import torch
from transformers import AutoTokenizer
from data_loader import load_and_process_all_data

def test_data_preparation():
    """Test data preparation and check for token sequence issues"""
    
    print("🔍 Data Preparation Debug Test")
    print("=" * 50)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load a small amount of data for testing
    print("\nLoading small dataset for testing...")
    try:
        train_data, val_data, data_stats = load_and_process_all_data(
            data_dir='data',
            ultrachat_samples=100,  # Small sample for testing
            wikipedia_samples=100,   # Small sample for testing
            generated_data_file="temp_generated_training_data.json",
            train_split=0.8,
            seed=42
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"\n📊 Dataset loaded successfully!")
    print(f"   Training tokens: {len(train_data):,}")
    print(f"   Validation tokens: {len(val_data):,}")
    
    # Test token sequences
    print(f"\n🔍 Testing token sequences...")
    
    # Check for repeated patterns
    sequence_length = 256
    tokens_per_batch = 48 * sequence_length
    
    print(f"   Checking sequences of length {sequence_length}...")
    
    # Sample a few sequences and check for patterns
    for i in range(0, min(len(train_data), tokens_per_batch * 3), tokens_per_batch):
        if i + tokens_per_batch > len(train_data):
            break
            
        sequence = train_data[i:i + tokens_per_batch]
        
        # Check for immediate repetitions
        repetitions = 0
        for j in range(len(sequence) - 1):
            if sequence[j] == sequence[j + 1]:
                repetitions += 1
        
        repetition_rate = repetitions / len(sequence) if len(sequence) > 0 else 0
        
        # Decode a sample
        sample_text = tokenizer.decode(sequence[:50], skip_special_tokens=True)
        
        print(f"\n   Sequence {i//tokens_per_batch + 1}:")
        print(f"      Repetition rate: {repetition_rate:.3f} ({repetitions}/{len(sequence)})")
        print(f"      Sample text: '{sample_text[:100]}...'")
        
        # Check for EOS token patterns
        eos_count = (sequence == tokenizer.eos_token_id).sum().item()
        print(f"      EOS tokens: {eos_count}")
        
        # Check for specific token patterns
        unique_tokens = len(set(sequence.tolist()))
        print(f"      Unique tokens: {unique_tokens}/{len(sequence)} ({unique_tokens/len(sequence):.3f})")
    
    # Test the training shift logic
    print(f"\n🔍 Testing training shift logic...")
    
    # Simulate the training loop logic
    for i in range(0, min(len(train_data), tokens_per_batch), tokens_per_batch):
        if i + tokens_per_batch + 1 > len(train_data):
            break
            
        input_tokens = train_data[i:i + tokens_per_batch]
        target_tokens = train_data[i + 1:i + tokens_per_batch + 1]
        
        # Check if targets are properly shifted
        if len(input_tokens) == len(target_tokens):
            # Decode a small sample to see the shift
            input_sample = tokenizer.decode(input_tokens[:10], skip_special_tokens=True)
            target_sample = tokenizer.decode(target_tokens[:10], skip_special_tokens=True)
            
            print(f"\n   Training sample {i//tokens_per_batch + 1}:")
            print(f"      Input:  '{input_sample}'")
            print(f"      Target: '{target_sample}'")
            
            # Check if target is just repeating the last input token
            if input_tokens[-1] == target_tokens[0]:
                print(f"      ⚠️  WARNING: Target starts with last input token!")
            
            break  # Just show one example
    
    print(f"\n✅ Data preparation test completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_data_preparation() 