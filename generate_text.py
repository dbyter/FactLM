#!/usr/bin/env python3
"""
FactLM Text Generation Script
Usage: python generate_text.py <model_path>

This script is compatible with both:
- Full models saved by model_trainer.py (with model_config included)
- Checkpoint files from training (best_model.pth, latest_checkpoint.pth, etc.)

The script automatically detects the model architecture and uses configurations
that match model_trainer.py defaults for consistency.
"""

import torch
import sys
import os
from transformers import AutoTokenizer
from factlm_model import FactLM

def get_default_model_config(vocab_size, d_model=None):
    """Get default model configuration matching model_trainer.py
    
    This ensures consistency between training and generation scripts.
    """
    # Single, efficient model configuration matching model_trainer.py
    return {
        'vocab_size': vocab_size,
        'hidden_size': 256,      # From model_trainer.py
        'num_layers': 6,         # From model_trainer.py  
        'dropout': 0.2,          # From model_trainer.py
        'd_model': 256,          # From model_trainer.py
        'max_len': 5000,
        'num_heads': 8           # From model_trainer.py (8 heads = 32-dim each)
    }


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
            
            # Advanced repetition penalty - much more aggressive
            if repetition_penalty != 1.0:
                generated_list = generated_tokens[0].tolist()
                
                # Apply very strong penalties for immediate repetitions
                for i, token_id in enumerate(generated_list):
                    # Distance from current position (recent tokens get higher penalty)
                    distance = len(generated_list) - i
                    
                    if distance <= 2:  # Last 2 tokens - prevent immediate repetition
                        penalty = repetition_penalty * 3.0  # Very strong penalty
                    elif distance <= 5:  # Very recent tokens
                        penalty = repetition_penalty * 2.0
                    elif distance <= 10:  # Recent tokens
                        penalty = repetition_penalty * 1.5
                    elif distance <= 20:  # Moderately recent tokens
                        penalty = repetition_penalty * 1.2
                    else:  # Older tokens
                        penalty = repetition_penalty
                    
                    logits[token_id] /= penalty
            
            # Enhanced N-gram repetition penalty - prevent repeated phrases
            if len(generated_tokens[0]) >= 4:
                generated_list = generated_tokens[0].tolist()
                
                # Check for immediate token repetition (strongest penalty)
                if len(generated_list) >= 1:
                    last_token = generated_list[-1]
                    logits[last_token] /= 5.0  # Very strong penalty for same token
                
                # Check for 2-gram, 3-gram, and 4-gram repetitions
                for n in [2, 3, 4, 5]:
                    if len(generated_list) >= n:
                        recent_ngram = tuple(generated_list[-n:])
                        
                        # Count ALL occurrences of this n-gram in the entire sequence
                        ngram_count = 0
                        for i in range(len(generated_list) - n + 1):
                            if tuple(generated_list[i:i+n]) == recent_ngram:
                                ngram_count += 1
                        
                        # Apply exponentially increasing penalties
                        if ngram_count > 1:  # If we've seen this n-gram before
                            # Look ahead: what tokens would complete this n-gram pattern?
                            for vocab_token_id in range(min(logits.size(0), 10000)):  # Limit for performance
                                # Check if adding this token would repeat the n-gram
                                test_ngram = recent_ngram[1:] + (vocab_token_id,)
                                
                                # Count how many times this new n-gram would appear
                                test_count = 0
                                for i in range(len(generated_list) - n + 1):
                                    if tuple(generated_list[i:i+n]) == test_ngram:
                                        test_count += 1
                                
                                if test_count > 0:
                                    # Exponential penalty based on n-gram size and frequency
                                    base_penalty = 2.0 + (n * 0.5)  # Larger n-grams get stronger penalties
                                    frequency_penalty = 1.0 + (test_count * 0.8)
                                    total_penalty = base_penalty * frequency_penalty
                                    logits[vocab_token_id] /= total_penalty
            
            # Additional anti-repetition: penalize tokens that create boring patterns
            if len(generated_tokens[0]) >= 6:
                generated_list = generated_tokens[0].tolist()
                
                # Check for alternating patterns (A-B-A-B...)
                if len(generated_list) >= 4:
                    # Look for A-B-A pattern and heavily penalize B
                    if (len(generated_list) >= 3 and 
                        generated_list[-3] == generated_list[-1]):
                        alternating_token = generated_list[-2]
                        logits[alternating_token] /= 4.0
                
                # Penalize tokens that appear too frequently in recent window
                recent_window_size = min(20, len(generated_list))
                recent_window = generated_list[-recent_window_size:]
                token_counts = {}
                for token in recent_window:
                    token_counts[token] = token_counts.get(token, 0) + 1
                
                # Apply frequency-based penalties
                for token_id, count in token_counts.items():
                    if count > 1:
                        frequency_penalty = 1.5 + (count * 0.3)
                        logits[token_id] /= frequency_penalty
            
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
            
            # Enhanced repetition detection - much more aggressive for quality
            if step > 2:  # Start checking very early for nonsense
                generated_list = generated_tokens[0].tolist()
                current_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Check for immediate token repetition (same token 2+ times in a row)
                if len(generated_list) >= 2:
                    if generated_list[-1] == generated_list[-2]:
                        print(f"   Stopping: Immediate token repetition detected")
                        break
                
                # Check for word salad - too many uncommon tokens in sequence
                if step > 5:
                    recent_tokens = generated_list[-5:]
                    decoded_recent = tokenizer.decode(recent_tokens, skip_special_tokens=True)
                    words = decoded_recent.split()
                    
                    # If we have multiple words that are very long or technical, likely word salad
                    long_technical_words = [w for w in words if len(w) > 8 and not any(c.isdigit() for c in w)]
                    if len(long_technical_words) >= 3:
                        print(f"   Stopping: Technical word salad detected")
                        break
                
                # Check for simple alternating pattern (A-B-A-B...)
                if len(generated_list) >= 4:
                    last_4 = generated_list[-4:]
                    if (last_4[0] == last_4[2] and last_4[1] == last_4[3] and 
                        last_4[0] != last_4[1]):
                        print(f"   Stopping: Alternating pattern detected")
                        break
                
                # Check for 3-token cycles (A-B-C-A-B-C...)
                if len(generated_list) >= 6:
                    last_6 = generated_list[-6:]
                    if (last_6[0] == last_6[3] and last_6[1] == last_6[4] and 
                        last_6[2] == last_6[5]):
                        print(f"   Stopping: 3-token cycle detected")
                        break
                
                # Check for exact phrase repetition (more aggressive)
                if len(generated_list) >= 4:
                    for phrase_len in [2, 3]:
                        if len(generated_list) >= phrase_len * 2:
                            recent_phrase = generated_list[-phrase_len:]
                            prev_phrase = generated_list[-phrase_len*2:-phrase_len]
                            if recent_phrase == prev_phrase:
                                print(f"   Stopping: {phrase_len}-token phrase repetition detected")
                                break
                
                # Check for low diversity in recent tokens (very aggressive)
                if step > 4:
                    recent_window_size = min(6, len(generated_list))
                    recent_window = generated_list[-recent_window_size:]
                    unique_ratio = len(set(recent_window)) / len(recent_window)
                    if unique_ratio < 0.7:  # Very high threshold for short sequences
                        print(f"   Stopping: Low diversity ({unique_ratio:.2f}) in recent tokens")
                        break
                
                # Stop on natural sentence endings much more aggressively
                if step > 3:
                    if any(current_text.rstrip().endswith(punct) for punct in ['.', '!', '?']):
                        print(f"   Stopping: Natural sentence ending detected")
                        break
                
                # Stop if the text seems to be degrading into nonsense
                if step > 8:
                    words = current_text.lower().split()
                    if len(words) >= 4:
                        # Check if recent words are getting very long (likely nonsense)
                        recent_words = words[-3:]
                        avg_word_length = sum(len(w) for w in recent_words) / len(recent_words)
                        if avg_word_length > 12:  # Very long average word length
                            print(f"   Stopping: Text degrading into long nonsense words")
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
    
    # Handle different checkpoint formats
    if 'model_config' in checkpoint:
        # New format with explicit model_config
        model_config = checkpoint['model_config']
    else:
        # Old format - reconstruct model_config from model state dict
        print("⚠️  Checkpoint doesn't have model_config, reconstructing from model state...")
        model_state = checkpoint['model_state_dict']
        
        # Extract configuration from model state dict
        vocab_size = model_state['embedding.weight'].shape[0]
        d_model = model_state['embedding.weight'].shape[1]
        
        # Count number of layers by looking for encoder layer patterns
        num_layers = 0
        layer_keys = []
        for key in model_state.keys():
            if key.startswith('encoder_layers.') and key.endswith('.self_attn.q.weight'):
                layer_keys.append(key)
                layer_num = int(key.split('.')[1])
                num_layers = max(num_layers, layer_num + 1)
        
        print(f"   Found layer keys: {layer_keys}")
        print(f"   Detected {num_layers} layers")
        
        # Fallback if layer detection failed
        if num_layers == 0:
            print("   ⚠️  Layer detection failed, examining all keys...")
            encoder_keys = [k for k in model_state.keys() if k.startswith('encoder_layers.')]
            print(f"   All encoder keys: {encoder_keys[:10]}...")  # Show first 10
            
            # Try to extract from any encoder layer key
            if encoder_keys:
                # Get the highest layer number found
                layer_numbers = []
                for key in encoder_keys:
                    try:
                        layer_num = int(key.split('.')[1])
                        layer_numbers.append(layer_num)
                    except:
                        continue
                if layer_numbers:
                    num_layers = max(layer_numbers) + 1
                    print(f"   Fallback detected {num_layers} layers from key analysis")
            
            # Final fallback - use common default
            if num_layers == 0:
                num_layers = 8  # Common default
                print(f"   Using fallback default: {num_layers} layers")
        
        # Extract number of heads from attention weights
        # q weight shape is [d_model, d_model], so num_heads = d_model / head_dim
        if 'encoder_layers.0.self_attn.q.weight' in model_state:
            # Use consistent 8 heads for the new simplified model
            num_heads = 8   # Always 8 heads in the new configuration
        else:
            num_heads = 8  # Default: 8 heads
        
        # Extract hidden size from feed forward layers
        if 'encoder_layers.0.feed_forward.0.weight' in model_state:
            # The feed-forward layer goes from d_model to d_model*4, but hidden_size is separate
            ff_hidden_size = model_state['encoder_layers.0.feed_forward.0.weight'].shape[0]
            # Verify this is d_model * 4 as expected
            if ff_hidden_size == d_model * 4:
                # hidden_size parameter matches d_model in the new simplified config
                hidden_size = d_model
            else:
                # Fallback if unexpected architecture
                hidden_size = d_model
        else:
            # Match new simplified config: hidden_size = d_model
            hidden_size = d_model
        
        # Set consistent dropout for the simplified model
        dropout = 0.2   # Standard dropout for the new configuration
        
        model_config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'd_model': d_model,
            'max_len': 5000,  # Default max length
            'num_heads': num_heads
        }
        
        # Validate the reconstructed config by comparing with defaults
        default_config = get_default_model_config(vocab_size, d_model)
        
        # If reconstruction failed or seems inconsistent, use defaults
        if num_layers == 0 or d_model < 64 or num_heads < 1:
            print("   ⚠️  Config reconstruction failed, using default configuration")
            model_config = default_config
        else:
            # Print what we reconstructed
            print(f"   Reconstructed config: vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")
            print(f"   Architecture details: hidden_size={hidden_size}, dropout={dropout}")
            
            # Warn if reconstructed config differs significantly from expected defaults
            if (d_model == default_config['d_model'] and 
                (num_layers != default_config['num_layers'] or 
                 num_heads != default_config['num_heads'])):
                print(f"   ⚠️  Config differs from current defaults - this may be from an older model version")
                print(f"   Default would be: layers={default_config['num_layers']}, heads={default_config['num_heads']}")
        
        print(f"   Final config: d_model={model_config['d_model']}, layers={model_config['num_layers']}, heads={model_config['num_heads']}")
    
    # Initialize model with configuration
    model = FactLM(**model_config)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded from: {model_path}")
    
    # Handle different checkpoint metadata formats
    if 'timestamp_utc' in checkpoint:
        print(f"   - Trained on: {checkpoint['timestamp_utc']}")
    elif 'epoch' in checkpoint:
        print(f"   - Checkpoint from epoch: {checkpoint['epoch']}")
    
    if 'device_used' in checkpoint:
        print(f"   - Device used: {checkpoint['device_used']}")
    
    vocab_size_from_checkpoint = checkpoint.get('vocab_size', model_config['vocab_size'])
    print(f"   - Vocabulary size: {vocab_size_from_checkpoint:,}")
    
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
    
    # Single balanced configuration for all generation - optimized for quality
    generation_config = {
        "temperature": 0.6,  # Increased for more natural variation (was 0.2)
        "repetition_penalty": 2.2,  # Much stronger anti-repetition (was 1.5)
        "top_k": 25,  # More selective token choice (was 40)
        "top_p": 0.85  # Slightly more focused (was 0.9)
    }
    
    for prompt in test_prompts[:3]:  # Test first 3 prompts
        print(f"\nPrompt: '{prompt}'")
        
        generated = generate_text(
            model=model,
            start_string=prompt,
            max_length=25,  # Much shorter for better quality (was 40)
            temperature=generation_config["temperature"],
            repetition_penalty=generation_config["repetition_penalty"],
            top_k=generation_config["top_k"],
            top_p=generation_config["top_p"],
            tokenizer=tokenizer
        )
        print(f"  Generated: {generated}")
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("💬 Interactive mode (type 'quit' to exit):")
    print(f"Using: temp={generation_config['temperature']}, rep_penalty={generation_config['repetition_penalty']}, top_k={generation_config['top_k']}, top_p={generation_config['top_p']}")
    print("="*60)
    
    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not prompt:
                continue
            
            print(f"\nGenerating response for: '{prompt}'")
            
            generated = generate_text(
                model=model,
                start_string=prompt,
                max_length=35,  # Shorter for better quality (was 60)
                temperature=generation_config["temperature"],
                repetition_penalty=generation_config["repetition_penalty"],
                top_k=generation_config["top_k"],
                top_p=generation_config["top_p"],
                tokenizer=tokenizer
            )
            print(f"  Response: {generated}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error generating text: {e}")


if __name__ == "__main__":
    main() 