# FactLM - A Transformer-Based Language Model

A PyTorch implementation of a transformer-based language model trained on Project Gutenberg books and UltraChat conversational data.

## Features

- **Transformer Architecture**: Multi-head attention with positional encoding
- **Multi-Source Training**: Combines Project Gutenberg books with UltraChat conversations
- **Automatic Book Processing**: Loads and processes multiple `book*.txt` files from the data directory
- **UltraChat Integration**: Automatically loads and processes conversational data from Hugging Face
- **Hugging Face Integration**: Uses GPT-2 tokenizer for professional text tokenization
- **GPU Support**: Supports CUDA, MPS (Apple Silicon), and CPU training
- **Advanced Text Generation**: Generate text with controllable temperature sampling and repetition penalties

## Architecture

The model consists of:
- **Embedding Layer**: Token embeddings with positional encoding
- **Encoder Layers**: Multi-head self-attention with feed-forward networks
- **Output Layer**: Linear projection to vocabulary size with log-softmax

## Requirements

```bash
pip install torch transformers datasets
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Add your Project Gutenberg books as `book1.txt`, `book2.txt`, etc. in the `data/` directory
2. Run the training script:

```bash
python model_trainer.py
```

The script will automatically:
- Find all `book*.txt` files in the data directory
- Load and process UltraChat conversational data from Hugging Face
- Clean and tokenize both text sources
- Combine and shuffle the training data
- Split into training/validation sets (80/20)
- Train the model for the specified epochs
- Save checkpoints every 5 epochs (configurable)
- Save the trained model with comprehensive metadata

### Checkpoint Management

The training process automatically saves checkpoints:
- **Regular checkpoints**: Every 5 epochs (by default)
- **Best model checkpoint**: Whenever validation loss improves
- **Latest checkpoint**: Always maintained for resuming training

Checkpoints are saved in timestamped directories under `checkpoints/` and include:
- Model state (weights and biases)
- Optimizer state (momentum, learning rate schedule)
- Training progress (epoch, losses, step count)

To resume training from a checkpoint, you can modify the training script to load from `checkpoints/training_TIMESTAMP/latest_checkpoint.pth`.

#### Resuming Training Example

```python
# After creating model, optimizer, and scheduler, load checkpoint
resume_info = load_checkpoint(
    checkpoint_path="checkpoints/training_20250115_143022_UTC/latest_checkpoint.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)

# Continue training from the loaded epoch
if resume_info:
    start_epoch = resume_info['epoch']
    best_val_loss = resume_info['best_val_loss']
```

### Training Data Sources

The model is trained on two types of data:

1. **Project Gutenberg Books**: Classic literature provides rich language patterns and vocabulary
2. **UltraChat Conversations**: Modern conversational data improves question-answering and dialogue capabilities

The training script automatically balances these sources and provides detailed statistics about the data composition.

### Model Configuration

Key parameters in `model_trainer.py`:
- `vocab_size`: 50,257 (GPT-2 tokenizer)
- `d_model`: 1024 (embedding dimension, large model)
- `num_layers`: 12 (transformer layers, increased for complexity)
- `num_heads`: 16 (attention heads, 64-dim each)
- `hidden_size`: 1024 (feed-forward layer size)
- `epochs`: 25 (optimized for larger model and dataset)
- `batch_size`: 16 (adjusted for larger model memory requirements)
- `learning_rate`: 0.00015 (conservative for large model stability)

### Large Model Performance

The enhanced model features:
- **4x larger embedding dimension** (1024 vs 256)
- **50% more transformer layers** (12 vs 8) 
- **2x more attention heads** (16 vs 8)
- **Expanded training data** (50K UltraChat conversations)

**Memory Requirements:**
- Recommended: 16GB+ GPU memory for training
- Minimum: 12GB GPU memory (with potential OOM risk)
- CPU fallback available but significantly slower

**Performance Benefits:**
- Better understanding of complex topics
- Improved coherence in longer text generation
- Enhanced conversational abilities
- More diverse and creative outputs

### UltraChat Configuration

You can customize the UltraChat data loading:

```python
ultrachat_data = load_ultrachat_data(
    dataset_name="stingning/ultrachat",
    num_samples=50000,  # Expanded dataset for better performance
    seed=42            # For reproducible sampling
)
```

### Text Generation

The model supports advanced text generation with multiple sampling strategies:

```python
generated = generate_text(
    model=trained_model,
    start_string="Your prompt here",
    max_length=100,
    temperature=0.8,
    tokenizer=tokenizer,
    repetition_penalty=1.2,
    top_k=50,
    top_p=0.9
)
```

## File Structure

```
FactLM/
├── factlm_model.py      # Model architecture
├── model_trainer.py     # Training script with UltraChat integration
├── generate_text.py     # Text generation script
├── data/
│   ├── book1.txt       # Training text files
│   └── book2.txt
├── models/             # Saved models directory
├── checkpoints/        # Training checkpoints directory
├── requirements.txt    # Dependencies
└── README.md
```

## Training Process

1. **Book Data Loading**: Automatically discovers and loads all `book*.txt` files
2. **UltraChat Loading**: Downloads and samples conversations from Hugging Face
3. **Text Processing**: Cleans formatting and converts conversations to training format
4. **Data Combination**: Merges and shuffles both data sources
5. **Tokenization**: Uses Hugging Face GPT-2 tokenizer for subword tokenization
6. **Training**: Next-token prediction with cross-entropy loss
7. **Validation**: Monitors validation loss with early stopping
8. **Model Saving**: Saves model with comprehensive metadata including data sources

## Model Details

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW with weight decay and beta parameters
- **Learning Rate Schedule**: Warmup followed by cosine decay
- **Attention Heads**: 16 heads per layer (64 dimensions each)
- **Model Size**: ~85M parameters (large architecture)
- **Embedding Dimension**: 1024 (d_model)
- **Feed-Forward Size**: 4096 (d_model * 4)
- **Positional Encoding**: Sinusoidal encoding up to 5000 positions
- **Dropout**: 0.15 for regularization
- **Gradient Clipping**: Max norm of 1.0

## Data Format

### Book Data
Books are processed by removing Project Gutenberg headers/footers and cleaning formatting.

### UltraChat Data
Conversations are formatted as:
```
Human: How can cross training benefit runners?
Assistant: Cross training can benefit runners in several ways: 1) Reduces injury risk by working different muscle groups, 2) Improves overall fitness and strength, 3) Prevents training monotony, 4) Enhances performance through balanced development.
```

Each conversation is separated by `<|endofconversation|>` markers and combined with book text for diverse training data.

## Sample Output

The model generates text that combines the narrative style of classic literature with modern conversational abilities:

```
Prompt: "The question of justice"
Generated: The question of justice and the rights of man are not to be determined by the will of the majority alone, but must consider the fundamental principles that govern both individual liberty and collective responsibility...

Prompt: "How can I improve my running?"
Generated: To improve your running performance, consider incorporating cross-training exercises such as strength training and plyometrics. These activities will help build the muscle groups that support your running form while reducing the risk of overuse injuries...
```

## Contributing

Feel free to submit issues and enhancement requests! Some areas for improvement:
- Additional dataset integration (e.g., other conversational datasets)
- Model architecture optimizations
- Advanced generation techniques
- Performance benchmarking

## License

This project is open source and available under the MIT License.