# FactLM - A Transformer-Based Language Model

A PyTorch implementation of a transformer-based language model trained on Project Gutenberg books.

## Features

- **Transformer Architecture**: Multi-head attention with positional encoding
- **Automatic Book Processing**: Loads and processes multiple `book*.txt` files from the data directory
- **Hugging Face Integration**: Uses GPT-2 tokenizer for professional text tokenization
- **GPU Support**: Supports CUDA, MPS (Apple Silicon), and CPU training
- **Text Generation**: Generate text with controllable temperature sampling

## Architecture

The model consists of:
- **Embedding Layer**: Token embeddings with positional encoding
- **Encoder Layers**: Multi-head self-attention with feed-forward networks
- **Output Layer**: Linear projection to vocabulary size with log-softmax

## Requirements

```bash
pip install torch transformers
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
- Clean and tokenize the text
- Split into training/validation sets (80/20)
- Train the model for the specified epochs
- Generate sample text with different prompts

### Model Configuration

Key parameters in `model_trainer.py`:
- `vocab_size`: 50,257 (GPT-2 tokenizer)
- `d_model`: 512 (embedding dimension)
- `num_layers`: 4 (transformer layers)
- `hidden_size`: 256
- `epochs`: 10
- `batch_size`: 64
- `learning_rate`: 0.0005

### Text Generation

The model supports controllable text generation with temperature sampling:

```python
generated = generate_text(
    model=trained_model,
    start_string="Your prompt here",
    max_length=50,
    temperature=0.8,
    tokenizer=tokenizer
)
```

## File Structure

```
FactLM/
├── factlm_model.py      # Model architecture
├── model_trainer.py     # Training and generation script
├── data/
│   ├── book1.txt       # Training text files
│   └── book2.txt
└── README.md
```

## Training Process

1. **Data Loading**: Automatically discovers and loads all `book*.txt` files
2. **Text Cleaning**: Removes Project Gutenberg headers/footers and cleans formatting
3. **Tokenization**: Uses Hugging Face GPT-2 tokenizer for subword tokenization
4. **Training**: Next-token prediction with negative log-likelihood loss
5. **Validation**: Monitors validation loss to track training progress
6. **Generation**: Tests learned patterns with multiple prompts

## Model Details

- **Loss Function**: Negative Log-Likelihood (NLLLoss)
- **Optimizer**: Adam with learning rate 0.0005
- **Attention Heads**: 8 heads per layer
- **Positional Encoding**: Sinusoidal encoding up to 5000 positions
- **Dropout**: 0.1 for regularization

## Sample Output

The model generates text in the style of the training books:

```
Prompt: "The question of justice"
Generated: The question of justice and the rights of man are not to be determined by the will of the majority...
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 