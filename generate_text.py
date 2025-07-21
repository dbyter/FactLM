#!/usr/bin/env python3
"""
FactLM Text Generation Script
Usage: python generate_text.py <model_path>
"""

import torch
import sys
import os
from transformers import AutoTokenizer
from factlm_model import FactLM

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


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_text.py <model_path>")
        print("\nExample:")
        print("  python generate_text.py models/factlm_model_20250721_021548_UTC.pth")
        print("\nAvailable models:")
        
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
            for model_file in sorted(model_files):
                print(f"  - models/{model_file}")
        else:
            print("  No models directory found!")
        return
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model, checkpoint = load_saved_model(model_path, tokenizer)
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA (NVIDIA GPU)"
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
        device_name = "MPS (Apple Silicon)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    model.to(device)
    print(f"Using device: {device} ({device_name})")
    
    # Predefined test prompts
    test_prompts = [
        "The nature of truth",
        "In matters of justice",
        "Society must consider",
        "What is the meaning of",
        "Where do we find",
        "How can we understand",
        "The question of morality"
    ]
    
    print("\n" + "="*60)
    print("FACTLM TEXT GENERATION")
    print("="*60)
    
    # Test with predefined prompts
    print("\n🤖 Testing with predefined prompts:")
    
    sampling_configs = [
        {"temperature": 0.8, "repetition_penalty": 1.3, "top_k": 50, "top_p": 0.9, "name": "Balanced"},
        {"temperature": 1.1, "repetition_penalty": 1.5, "top_k": 40, "top_p": 0.85, "name": "Creative"}, 
        {"temperature": 0.6, "repetition_penalty": 1.2, "top_k": 30, "top_p": 0.95, "name": "Conservative"}
    ]
    
    for prompt in test_prompts[:3]:  # Test first 3 prompts
        print(f"\nPrompt: '{prompt}'")
        
        for config in sampling_configs:
            generated = generate_text(
                model=model,
                start_string=prompt,
                max_length=40,
                temperature=config["temperature"],
                repetition_penalty=config["repetition_penalty"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                tokenizer=tokenizer
            )
            print(f"  {config['name']:12}: {generated}")
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("💬 Interactive mode (type 'quit' to exit):")
    print("="*60)
    
    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not prompt:
                continue
            
            print(f"\nGenerating responses for: '{prompt}'")
            
            for config in sampling_configs:
                generated = generate_text(
                    model=model,
                    start_string=prompt,
                    max_length=75,
                    temperature=config["temperature"],
                    repetition_penalty=config["repetition_penalty"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    tokenizer=tokenizer
                )
                print(f"  {config['name']:12}: {generated}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error generating text: {e}")


if __name__ == "__main__":
    main() 