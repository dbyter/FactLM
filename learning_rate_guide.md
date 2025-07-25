# Learning Rate Optimization Guide for FactLM

## Current Changes Made

### 1. **Training Learning Rate** (model_trainer.py)
- **Old**: `0.00015` (very conservative)
- **New**: `0.001` (6.7x higher)
- **Available options**: 0.0005, 0.001, 0.002

### 2. **Fine-tuning Learning Rate** (model_fine_tuner.py)
- **Old**: `1e-5` (extremely conservative)
- **New**: `5e-4` (50x higher!)
- **Available options**: 5e-5, 1e-4, 5e-4, 1e-3

### 3. **Optimizer Improvements**
- **Beta2**: Changed from 0.999 → 0.95 (faster adaptation)
- **Warmup**: Reduced from 1/2 epoch → 1/4 epoch (faster learning)
- **Patience**: Reduced from 8 → 5 epochs (faster convergence detection)
- **Min LR**: Reduced from 10% → 5% of base LR (more aggressive decay)

## Why These Changes Help Convergence

### 🚀 **Higher Learning Rates**
- **Faster parameter updates**: Escape shallow local minima
- **Better exploration**: Find optimal parameter regions quicker
- **Reduced training time**: Reach convergence in fewer epochs

### ⚡ **Faster Warmup**
- **Quick stabilization**: Model adapts faster to data distribution
- **Less conservative**: Avoids overly slow initial learning

### 🎯 **Improved Optimizer**
- **Lower beta2** (0.95): Faster adaptation to recent gradients
- **Reduced patience**: Stop training sooner if not improving

## Testing Different Learning Rates

### For Training (model_trainer.py)
```python
# Line ~540: Uncomment one of these options:
# learning_rate = 0.0005   # Conservative but faster than original
learning_rate = 0.001     # Recommended starting point
# learning_rate = 0.002    # Aggressive - use if 0.001 works well
```

### For Fine-tuning (model_fine_tuner.py)
```python
# Default is now 5e-4, but you can try:
# learning_rate = 1e-4     # More conservative
learning_rate = 5e-4     # Current default (recommended)
# learning_rate = 1e-3     # More aggressive
```

## Signs of Good Convergence

### ✅ **Training is working well if you see:**
- Loss decreasing consistently over epochs
- Validation loss following training loss (not diverging)
- Gradient norms between 0.1-2.0 (stable)
- Learning rate schedule working (LR decreasing over time)

### ⚠️ **Warning signs:**
- Loss oscillating wildly → reduce learning rate
- Gradient norms > 5.0 → reduce learning rate or increase clipping
- Validation loss increasing while train loss decreases → overfitting

## Quick Convergence Tests

### 1. **Start with recommended settings** (already set)
```bash
python model_trainer.py  # Uses LR=0.001
```

### 2. **If training is unstable**, reduce LR:
Edit model_trainer.py line ~540:
```python
learning_rate = 0.0005  # More conservative
```

### 3. **If training is too slow**, increase LR:
```python
learning_rate = 0.002   # More aggressive
```

### 4. **Monitor early epochs closely**
- Check loss at epochs 1-3
- If loss doesn't drop significantly → increase LR
- If loss explodes → decrease LR

## Expected Timeline with New Settings

### Training (model_trainer.py)
- **Epochs 1-3**: Rapid initial loss decrease
- **Epochs 4-10**: Steady improvement
- **Epochs 11-25**: Gradual refinement

### Fine-tuning (model_fine_tuner.py)
- **Epoch 1**: Significant adaptation to conversational style
- **Epochs 2-3**: Refinement and stabilization

## Troubleshooting

### Problem: "Loss not decreasing fast enough"
**Solution**: Increase learning rate by 2-3x

### Problem: "Loss oscillating or exploding"
**Solution**: Decrease learning rate by 2-3x, check gradient clipping

### Problem: "Validation loss worse than training"
**Solution**: 
- Add more regularization (increase dropout)
- Reduce learning rate slightly
- Check for data leakage

### Problem: "Training stops improving early"
**Solution**:
- Increase learning rate
- Reduce warmup steps
- Check if model has enough capacity

## Quick Learning Rate Finder

Run a short training session (5 epochs) with different rates:

```python
# Test these learning rates for 5 epochs each:
test_rates = [0.0005, 0.001, 0.002, 0.003]

# Use the one where:
# 1. Loss decreases fastest in first 2 epochs
# 2. Training remains stable (no explosions)
# 3. Validation loss follows training loss
```

## Summary

The new settings should give you **much faster convergence**:
- **Training**: 6.7x higher learning rate
- **Fine-tuning**: 50x higher learning rate  
- **Optimizer**: Faster adaptation and more aggressive schedules

Try the current settings first - they should show dramatic improvement in convergence speed! 