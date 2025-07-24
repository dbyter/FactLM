# FactLM Training Data Generation

This document explains how to use the new training data generation system that creates custom datasets using GPT-4o-mini.

## 🎯 Overview

The training data generation system consists of several components:

1. **`generate_training_data.py`** - Main script that generates prompts and queries GPT-4o-mini
2. **`data_loader.py`** - Enhanced with support for generated training data  
3. **`test_prompt_generation.py`** - Test script to preview prompt variety
4. **`example_generate_data.py`** - Example workflow and usage demonstration

## 🚀 Quick Start

### 1. Setup OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or add to your shell profile:
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Test Prompt Generation (No API calls)

```bash
python3 test_prompt_generation.py
```

This shows you the variety of prompts that will be generated without using any API credits.

### 3. Generate Example Dataset (50 prompts)

```bash
python3 example_generate_data.py
```

This creates a small example dataset and shows how it integrates with your existing data loader.

### 4. Generate Full Dataset (10,000 prompts)

```bash
python3 example_generate_data.py full
```

Or directly:
```bash
python3 generate_training_data.py
```

## 📊 What Gets Generated

### Prompt Categories

The system generates diverse prompts across 10 categories:

1. **General Knowledge** - "What is {topic} and why is it important?"
2. **Reasoning** - "What factors should be considered when evaluating {topic}?"
3. **Creative Writing** - "Write a short story about {topic}."
4. **Problem Solving** - "How would you solve this problem: {problem}?"
5. **Explanation** - "Explain {concept} to someone who has never heard of it."
6. **Analysis** - "What are the pros and cons of {topic}?"
7. **Comparison** - "Compare and contrast {item1} and {item2}."
8. **Instruction Following** - "Please list the steps to {task}."
9. **Conversation** - "What advice would you give to someone struggling with {challenge}?"
10. **Factual Questions** - "What are some interesting facts about {topic}?"

### Topics Covered

49+ topics including:
- **Technology**: AI, machine learning, cybersecurity, blockchain, quantum computing
- **Science**: Physics, chemistry, biology, neuroscience, climate change
- **Skills**: Programming, web development, languages, fitness, cooking
- **Academic**: Mathematics, economics, psychology, philosophy, literature

## 📁 Output Format

Generated data is saved as JSON in a format compatible with your existing data loader:

```json
[
  {
    "data": ["Human prompt here", "GPT-4o-mini response here"],
    "metadata": {
      "timestamp": "2024-01-01T12:00:00",
      "model": "gpt-4o-mini",
      "source": "generated_training_data"
    }
  }
]
```

## 🔧 Configuration Options

### In `generate_training_data.py`:

```python
# Default settings
TOTAL_PROMPTS = 10000  # Number of prompts to generate
BATCH_SIZE = 100       # Prompts per batch (for rate limiting)
OUTPUT_FILE = "generated_training_data.json"
```

### In `data_loader.py`:

The enhanced `load_and_process_all_data()` function now accepts:

```python
train_data, val_data, stats = load_and_process_all_data(
    data_dir='data',                              # Book files directory
    ultrachat_samples=50000,                      # UltraChat conversations
    generated_data_file="generated_training_data.json",  # Your generated data
    train_split=0.8,                              # Train/validation split
    seed=42                                       # Random seed
)
```

## 📈 Data Integration

Your final training dataset will combine:

1. **📚 Books** - Project Gutenberg texts (clean literature)
2. **💬 UltraChat** - Conversational AI training data
3. **🤖 Generated** - Custom GPT-4o-mini responses

The data loader automatically:
- Tokenizes all data sources consistently
- Combines them preserving text structure
- Splits into training and validation sets
- Provides detailed statistics

## 💡 Usage Examples

### Basic Generation

```python
from generate_training_data import TrainingDataGenerator

generator = TrainingDataGenerator()
results = generator.generate_training_dataset(
    total_prompts=1000,
    batch_size=50,
    output_file="my_training_data.json"
)
```

### Using Generated Data

```python
from data_loader import load_and_process_all_data

train_data, val_data, stats = load_and_process_all_data(
    generated_data_file="my_training_data.json"
)

print(f"Total tokens: {stats['total_tokens']:,}")
print(f"Generated tokens: {stats['generated_tokens']:,}")
```

## ⚠️ Important Notes

### API Costs
- GPT-4o-mini is cost-effective but still uses credits
- 10,000 prompts ≈ 5M tokens (varies by response length)
- Monitor your OpenAI usage dashboard

### Rate Limits
- Built-in delays between requests (0.1s success, 1.0s error)
- Batch processing reduces rate limit issues
- Automatic error handling and retry logic

### Data Quality
- Responses are capped at 500 tokens each
- Temperature set to 0.7 for good variety
- System prompt encourages clear, informative responses

## 🛠️ Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

2. **Rate limit errors**
   - Reduce batch size
   - The script will automatically retry with delays

3. **Memory issues with large datasets**
   - The data loader chunks large texts automatically
   - Consider reducing `ultrachat_samples` if needed

4. **Import errors**
   ```bash
   pip install -r requirements.txt
   ```

### Debugging

- Raw results are saved to `*_raw.json` files for debugging
- Intermediate saves every 5 batches prevent data loss
- Detailed progress reporting throughout generation

## 🎉 Next Steps

After generating your training data:

1. **Verify the data**: Check the statistics and sample conversations
2. **Integrate with training**: Use the tensors from the data loader
3. **Monitor quality**: Review some generated conversations manually
4. **Iterate**: Adjust prompt templates or topics based on your needs

## 📝 Files Created

After running the system, you'll have:

- `generated_training_data.json` - Main training data
- `generated_training_data_raw.json` - Raw results with metadata
- `temp_*.json` - Intermediate saves (during generation)
- `example_training_data.json` - Small example dataset

Happy training! 🚀 