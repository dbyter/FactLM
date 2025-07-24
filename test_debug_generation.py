#!/usr/bin/env python3
"""
Debug Generation Test Script

This script demonstrates token-level debugging for text generation.
Run this to see detailed token selection at each step.

Usage: python test_debug_generation.py [model_path]
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factlm_model import FactLM
from generate_text import generate_text, load_saved_model, get_default_model_config

def test_debug_generation():
    """Test debug generation with a simple example"""
    
    print("🔍 FactLM Debug Generation Test")
    print("=" * 50)
    
    # Check if model path was provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
    else:
        # Look for any available model in models directory
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
            if model_files:
                model_path = os.path.join("models", sorted(model_files)[-1])  # Use most recent
                print(f"📂 Using model: {model_path}")
            else:
                print("❌ No trained models found in models/ directory")
                print("   Please train a model first with: python model_trainer.py")
                return
        else:
            print("❌ No models directory found")
            print("   Please train a model first with: python model_trainer.py")
            return
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    try:
        model, checkpoint = load_saved_model(model_path, tokenizer)
        
        # Auto-detect device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "CUDA"
        elif torch.backends.mps.is_available():
            device = torch.device("mps") 
            device_name = "MPS"
        else:
            device = torch.device("cpu")
            device_name = "CPU"
        
        model.to(device)
        print(f"Using device: {device_name}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test prompts for debugging
    test_prompts = [
        "What is artificial intelligence",
        "The nature of truth",
        "Hello, how are you"
    ]
    
    print(f"\n🔬 DEBUG ANALYSIS")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}/3: '{prompt}'")
        print("-" * 40)
        
        try:
            generated = generate_text(
                model=model,
                start_string=prompt,
                max_length=15,  # Short for detailed analysis
                temperature=0.8,
                repetition_penalty=1.1,
                top_k=50,
                top_p=0.9,
                tokenizer=tokenizer,
                debug=True
            )
            
            print(f"\n✅ Test {i} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            continue
        
        if i < len(test_prompts):
            input("\n⏸️  Press Enter to continue to next test...")
    
    print(f"\n🎉 Debug testing completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_debug_generation() 