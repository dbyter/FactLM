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


def generate_text(model, start_string, max_length, temperature=0.8, tokenizer=None, repetition_penalty=1.2, top_k=50, top_p=0.9, debug=False):
    """Generate text using the trained model with advanced sampling and anti-repetition"""
    model.eval()
    device = next(model.parameters()).device
    
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text generation")
    
    if debug:
        print(f"\n🔍 DEBUG MODE: Generating text for '{start_string}'")
        print(f"   Settings: temp={temperature}, rep_penalty={repetition_penalty}, top_k={top_k}, top_p={top_p}")
        print("=" * 80)
    
    # Tokenize the start string
    tokens = tokenizer.encode(start_string, add_special_tokens=True)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = input_ids.clone()
    
    if debug:
        print(f"📝 Input tokens: {tokens}")
        print(f"📝 Input decoded: '{tokenizer.decode(tokens)}'")
        print("=" * 80)
    
    with torch.no_grad():
        for step in range(max_length):
            # Use only recent context to avoid memory issues
            context_length = min(generated_tokens.size(1), 512)  # Match training sequence length
            context = generated_tokens[:, -context_length:]
            
            outputs = model(context)
            logits = outputs[0, -1, :].clone()
            
            if debug:
                print(f"\n🔸 Step {step + 1}:")
                print(f"   Context length: {context.size(1)} tokens")
                print(f"   Generated so far: '{tokenizer.decode(generated_tokens[0], skip_special_tokens=True)}'")
            
            # Simple repetition penalty
            if repetition_penalty != 1.0:
                generated_list = generated_tokens[0].tolist()
                
                # Apply standard repetition penalty to previously generated tokens
                for token_id in set(generated_list):
                    logits[token_id] /= repetition_penalty
            
            # Store original logits for debugging
            if debug:
                original_logits = logits.clone()
            
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
            
            # Get top 5 tokens for debugging
            if debug:
                # Before sampling
                probs_before_sampling = torch.softmax(logits, dim=-1)
                top5_probs, top5_indices = torch.topk(probs_before_sampling, 5)
                
                print(f"   🏆 Top 5 candidates:")
                for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
                    token_text = tokenizer.decode([idx.item()], skip_special_tokens=True)
                    # Handle special tokens and weird characters
                    if not token_text.strip():
                        token_text = f"<TOKEN_{idx.item()}>"
                    original_logit = original_logits[idx.item()].item()
                    final_logit = logits[idx.item()].item()
                    print(f"      {i+1}. '{token_text}' (ID: {idx.item()}) - Prob: {prob.item():.4f} | Logit: {original_logit:.2f} → {final_logit:.2f}")
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            if debug:
                chosen_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                if not chosen_token_text.strip():
                    chosen_token_text = f"<TOKEN_{next_token.item()}>"
                chosen_prob = probs[next_token.item()].item()
                print(f"   ✅ CHOSEN: '{chosen_token_text}' (ID: {next_token.item()}) - Prob: {chosen_prob:.4f}")
            
            # Stop conditions
            if next_token.item() == tokenizer.eos_token_id:
                if debug:
                    print(f"   🛑 Stopping: EOS token detected")
                break
            
            # Enhanced repetition detection - much more aggressive for quality
            if step > 2:  # Start checking very early for nonsense
                generated_list = generated_tokens[0].tolist()
                current_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Check for immediate token repetition (same token 3+ times in a row)
                if len(generated_list) >= 3:
                    if generated_list[-1] == generated_list[-2] == generated_list[-3]:
                        if debug:
                            print(f"   🛑 Stopping: Immediate token repetition detected")
                        break
                
                # Stop on natural sentence endings (more permissive)
                if step > 10:
                    if any(current_text.rstrip().endswith(punct) for punct in ['.', '!', '?']):
                        if debug:
                            print(f"   🛑 Stopping: Natural sentence ending detected")
                        break
            
            # Add token to sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
            
            # Simple sentence ending detection (less aggressive)
            if step > 20:  # Only after generating substantial text
                current_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Check if we just completed a sentence and have enough content
                if any(current_text.rstrip().endswith(punct) for punct in ['.', '!', '?']):
                    if len(current_text.split()) >= 10:  # At least 10 words
                        if debug:
                            print(f"   🛑 Stopping: Complete sentence with enough content")
                        break
    
    # Decode the generated tokens
    generated_tokens = generated_tokens[0].cpu().tolist()
    final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    if debug:
        print("=" * 80)
        print(f"🎯 FINAL RESULT: '{final_text}'")
        print(f"📊 Total tokens generated: {len(generated_tokens) - len(tokens)}")
        print("=" * 80)
    
    return final_text


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
        
        # First, check if this is a legacy (custom architecture) or new (PyTorch TransformerEncoder) checkpoint
        has_legacy_keys = any(key.startswith('encoder_layers.') for key in model_state.keys())
        has_new_keys = any(key.startswith('encoder.layers.') for key in model_state.keys())
        
        print(f"   Architecture detection:")
        print(f"   - Legacy custom architecture keys: {'Yes' if has_legacy_keys else 'No'}")
        print(f"   - PyTorch TransformerEncoder keys: {'Yes' if has_new_keys else 'No'}")
        
        if has_legacy_keys and not has_new_keys:
            print("   ❌ ERROR: This checkpoint uses the old custom architecture!")
            print("   The model code has been updated to use PyTorch's TransformerEncoder.")
            print("   This checkpoint is incompatible with the current model architecture.")
            print("")
            print("   🔧 SOLUTION: Please retrain your model with the updated architecture.")
            print("   The new architecture has these improvements:")
            print("   - Better stability and faster training")
            print("   - Proper causal attention masks") 
            print("   - More efficient PyTorch implementation")
            print("   - Improved weight initialization")
            print("")
            raise RuntimeError("Incompatible checkpoint format. Please retrain with updated architecture.")
        
        # Extract configuration from model state dict
        vocab_size = model_state['embedding.weight'].shape[0]
        d_model = model_state['embedding.weight'].shape[1]
        
        # Count number of layers by looking for encoder layer patterns
        num_layers = 0
        layer_keys = []
        
        # Look for PyTorch TransformerEncoder layer patterns
        for key in model_state.keys():
            if key.startswith('encoder.layers.') and '.self_attn.' in key:
                layer_keys.append(key)
                # Extract layer number from key like "encoder.layers.0.self_attn.in_proj_weight"
                try:
                    layer_num = int(key.split('.')[2])
                    num_layers = max(num_layers, layer_num + 1)
                except (IndexError, ValueError):
                    continue
        
        print(f"   Found layer keys: {layer_keys[:5]}...")  # Show first 5
        print(f"   Detected {num_layers} layers")
        
        # Fallback if layer detection failed
        if num_layers == 0:
            print("   ⚠️  TransformerEncoder layer detection failed, trying legacy patterns...")
            
            # Try legacy custom architecture patterns
            legacy_keys = [k for k in model_state.keys() if 'encoder_layers.' in k or 'self_attn.' in k]
            print(f"   Legacy keys found: {len(legacy_keys)}")
            
            if legacy_keys:
                # Try to extract from legacy encoder layer keys
                layer_numbers = []
                for key in legacy_keys:
                    try:
                        if 'encoder_layers.' in key:
                            layer_num = int(key.split('encoder_layers.')[1].split('.')[0])
                            layer_numbers.append(layer_num)
                    except:
                        continue
                
                if layer_numbers:
                    num_layers = max(layer_numbers) + 1
                    print(f"   Legacy detection found {num_layers} layers")
            
            # Final fallback - use current model default (6 layers)
            if num_layers == 0:
                num_layers = 6  # Match current model_trainer.py default
                print(f"   Using current model default: {num_layers} layers")
        
        # Extract number of heads from attention weights
        # q weight shape is [d_model, d_model], so num_heads = d_model / head_dim
        if 'encoder.layers.0.self_attn.in_proj_weight' in model_state:
            # PyTorch TransformerEncoder uses in_proj_weight (combined q,k,v)
            # Use consistent 8 heads for the new simplified model
            num_heads = 8   # Always 8 heads in the new configuration
        elif 'encoder_layers.0.self_attn.q.weight' in model_state:
            # Legacy custom architecture
            num_heads = 8   # Consistent with current config
        else:
            num_heads = 8  # Default: 8 heads
        
        # Extract hidden size from feed forward layers
        if 'encoder.layers.0.linear1.weight' in model_state:
            # PyTorch TransformerEncoder uses linear1/linear2
            # The feed-forward layer goes from d_model to d_model*4, but hidden_size is separate
            ff_hidden_size = model_state['encoder.layers.0.linear1.weight'].shape[0]
            # Verify this is d_model * 4 as expected
            if ff_hidden_size == d_model * 4:
                # hidden_size parameter matches d_model in the new simplified config
                hidden_size = d_model
            else:
                # Fallback if unexpected architecture
                hidden_size = d_model
        elif 'encoder_layers.0.feed_forward.0.weight' in model_state:
            # Legacy custom architecture
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
    
    # Multiple sampling configurations for different output styles
    sampling_configs = {
        "conservative": {
            "name": "Conservative (Best Quality)",
            "temperature": 1.2,  # Much higher to break deterministic patterns
            "repetition_penalty": 1.3,  # Slightly higher to prevent repetition
            "top_k": 40,
            "top_p": 0.85,
            "description": "Focused, high-quality, predictable outputs"
        },
        "balanced": {
            "name": "Balanced (Recommended)",
            "temperature": 1.5,  # Much higher to break deterministic patterns
            "repetition_penalty": 1.2,  # Higher to prevent repetition
            "top_k": 50,
            "top_p": 0.9,
            "description": "Good balance of quality and creativity"
        },
        "creative": {
            "name": "Creative (Experimental)",
            "temperature": 2.0,  # Very high to break deterministic patterns
            "repetition_penalty": 1.1,  # Higher to prevent repetition
            "top_k": 60,
            "top_p": 0.95,
            "description": "More diverse, creative, and experimental outputs"
        }
    }
    
    # Test with predefined prompts using all configurations
    print("\n🤖 Testing with predefined prompts (all sampling modes):")
    
    # First, do a debug test with a specific prompt
    debug_prompt = "What is artificial intelligence"
    print(f"\n🔍 DEBUG TEST: Token-level analysis for '{debug_prompt}'")
    print("=" * 60)
    
    debug_generated = generate_text(
        model=model,
        start_string=debug_prompt,
        max_length=20,  # Shorter for detailed debugging
        temperature=0.8,
        repetition_penalty=1.1,
        top_k=50,
        top_p=0.9,
        tokenizer=tokenizer,
        debug=True  # Enable debug mode
    )
    
    print(f"\n" + "="*60)
    print("🤖 Regular Testing (all sampling modes):")
    print("="*60)
    
    for config_name, config in sampling_configs.items():
        print(f"\n{'='*40}")
        print(f"🎯 {config['name']}")
        print(f"   {config['description']}")
        print(f"   Settings: temp={config['temperature']}, rep_penalty={config['repetition_penalty']}, top_k={config['top_k']}, top_p={config['top_p']}")
        print('='*40)
        
        for prompt in test_prompts[:2]:  # Test first 2 prompts per config
            print(f"\nPrompt: '{prompt}'")
            
            generated = generate_text(
                model=model,
                start_string=prompt,
                max_length=50,  # Increased from 25 to allow more complete thoughts
                temperature=config["temperature"],
                repetition_penalty=config["repetition_penalty"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                tokenizer=tokenizer
            )
            print(f"  Generated: {generated}")
    
    # Interactive mode with configuration selection
    print(f"\n{'='*60}")
    print("💬 Interactive mode (type 'quit' to exit):")
    print("🎛️  Available sampling modes:")
    for key, config in sampling_configs.items():
        print(f"   {key}: {config['name']} - {config['description']}")
    print("="*60)
    
    # Default to balanced mode
    current_config = "balanced"
    current_settings = sampling_configs[current_config]
    
    print(f"\n🎯 Current mode: {current_settings['name']}")
    print(f"💡 Type 'mode <conservative|balanced|creative>' to change sampling mode")
    print(f"💡 Type 'settings' to see current configuration")
    print(f"💡 Type 'debug <your prompt>' to see token-level generation details")
    
    while True:
        try:
            user_input = input("\nEnter your prompt (or command): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
            
            # Handle debug mode
            if user_input.lower().startswith('debug '):
                debug_prompt = user_input[6:].strip()  # Remove 'debug ' prefix
                if debug_prompt:
                    print(f"\n🔍 DEBUG MODE: Token-level analysis for '{debug_prompt}'")
                    print("=" * 60)
                    
                    debug_generated = generate_text(
                        model=model,
                        start_string=debug_prompt,
                        max_length=30,  # Reasonable length for debugging
                        temperature=current_settings["temperature"],
                        repetition_penalty=current_settings["repetition_penalty"],
                        top_k=current_settings["top_k"],
                        top_p=current_settings["top_p"],
                        tokenizer=tokenizer,
                        debug=True
                    )
                    continue
                else:
                    print("❌ Please provide a prompt after 'debug '. Example: debug What is AI?")
                    continue
            
            # Handle mode switching
            if user_input.lower().startswith('mode '):
                new_mode = user_input.lower().replace('mode ', '').strip()
                if new_mode in sampling_configs:
                    current_config = new_mode
                    current_settings = sampling_configs[current_config]
                    print(f"✅ Switched to {current_settings['name']}")
                    print(f"   {current_settings['description']}")
                    continue
                else:
                    print(f"❌ Unknown mode '{new_mode}'. Available: {', '.join(sampling_configs.keys())}")
                    continue
            
            # Handle settings display
            if user_input.lower() == 'settings':
                print(f"\n🎯 Current: {current_settings['name']}")
                print(f"   Temperature: {current_settings['temperature']}")
                print(f"   Repetition Penalty: {current_settings['repetition_penalty']}")
                print(f"   Top-K: {current_settings['top_k']}")
                print(f"   Top-P: {current_settings['top_p']}")
                print(f"   Description: {current_settings['description']}")
                continue
            
            # Generate text with current configuration
            print(f"\n🎯 Generating with {current_settings['name']} mode...")
            print(f"Prompt: '{user_input}'")
            
            generated = generate_text(
                model=model,
                start_string=user_input,
                max_length=60,  # Increased from 35 to allow more complete responses
                temperature=current_settings["temperature"],
                repetition_penalty=current_settings["repetition_penalty"],
                top_k=current_settings["top_k"],
                top_p=current_settings["top_p"],
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