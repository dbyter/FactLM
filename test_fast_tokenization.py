#!/usr/bin/env python3
"""
Test script to demonstrate the speed improvements of fast tokenization methods
"""

import time
import random
import string
from transformers import AutoTokenizer
from data_loader import tokenize_text_data, tokenize_text_data_batch, tokenize_text_data_fast

def generate_test_text(length=100000):
    """Generate random test text of specified length"""
    print(f"Generating {length:,} characters of test text...")
    
    # Generate random sentences
    sentences = []
    words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural', 'networks', 'transformer', 'model', 'training', 'data', 'processing']
    
    for _ in range(length // 50):  # Roughly 50 chars per sentence
        sentence_length = random.randint(5, 15)
        sentence_words = random.choices(words, k=sentence_length)
        sentence = ' '.join(sentence_words).capitalize() + '.'
        sentences.append(sentence)
    
    return ' '.join(sentences)

def benchmark_tokenization_methods(text, tokenizer):
    """Benchmark different tokenization methods"""
    print(f"\n🔍 Benchmarking tokenization methods on {len(text):,} characters...")
    print("=" * 60)
    
    # Test 1: Original method
    print("\n1️⃣ Testing original tokenization method...")
    start_time = time.time()
    try:
        tokens_original = tokenize_text_data(text, tokenizer)
        original_time = time.time() - start_time
        print(f"   ✅ Original method: {len(tokens_original):,} tokens in {original_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Original method failed: {e}")
        original_time = float('inf')
        tokens_original = []
    
    # Test 2: Fast parallel method
    print("\n2️⃣ Testing fast parallel tokenization method...")
    start_time = time.time()
    try:
        tokens_fast = tokenize_text_data_fast(text, tokenizer, batch_size=1000, max_workers=20)
        fast_time = time.time() - start_time
        print(f"   ✅ Fast parallel method: {len(tokens_fast):,} tokens in {fast_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Fast parallel method failed: {e}")
        fast_time = float('inf')
        tokens_fast = []
    
    # Test 3: Ultra-fast batch method
    print("\n3️⃣ Testing ultra-fast batch tokenization method...")
    start_time = time.time()
    try:
        tokens_batch = tokenize_text_data_batch(text, tokenizer, batch_size=1000)
        batch_time = time.time() - start_time
        print(f"   ✅ Ultra-fast batch method: {len(tokens_batch):,} tokens in {batch_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Ultra-fast batch method failed: {e}")
        batch_time = float('inf')
        tokens_batch = []
    
    # Results summary
    print(f"\n📊 Results Summary:")
    print("-" * 40)
    
    if original_time != float('inf'):
        print(f"Original method:     {original_time:.2f}s")
    if fast_time != float('inf'):
        speedup_fast = original_time / fast_time if original_time != float('inf') else 0
        print(f"Fast parallel:       {fast_time:.2f}s ({speedup_fast:.1f}x faster)")
    if batch_time != float('inf'):
        speedup_batch = original_time / batch_time if original_time != float('inf') else 0
        print(f"Ultra-fast batch:    {batch_time:.2f}s ({speedup_batch:.1f}x faster)")
    
    # Verify token counts are similar
    if tokens_original and tokens_fast and tokens_batch:
        print(f"\n🔍 Token count verification:")
        print(f"Original: {len(tokens_original):,} tokens")
        print(f"Fast:     {len(tokens_fast):,} tokens")
        print(f"Batch:    {len(tokens_batch):,} tokens")
        
        # Check if they're reasonably close (within 5%)
        if abs(len(tokens_original) - len(tokens_fast)) / len(tokens_original) < 0.05:
            print("✅ Fast method token count is similar")
        else:
            print("⚠️  Fast method token count differs significantly")
            
        if abs(len(tokens_original) - len(tokens_batch)) / len(tokens_original) < 0.05:
            print("✅ Batch method token count is similar")
        else:
            print("⚠️  Batch method token count differs significantly")

def main():
    """Main test function"""
    print("🚀 Fast Tokenization Benchmark Test")
    print("=" * 50)
    
    # Load tokenizer
    print("\n🔤 Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate test text
    test_text = generate_test_text(200000)  # 200K characters
    print(f"✅ Generated {len(test_text):,} characters of test text")
    
    # Run benchmarks
    benchmark_tokenization_methods(test_text, tokenizer)
    
    print(f"\n🎉 Benchmark complete!")
    print(f"\n💡 Tips for faster tokenization:")
    print(f"   - Use batch_size=1000-2000 for optimal performance")
    print(f"   - Using max_workers=20 for maximum parallel processing")
    print(f"   - The ultra-fast batch method is usually fastest")
    print(f"   - For very large texts, consider chunking first")
    print(f"   - 20 workers should provide 3-5x speedup over 4 workers")

if __name__ == "__main__":
    main() 