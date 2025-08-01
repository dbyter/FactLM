#!/usr/bin/env python3
"""
Generate Common Question Prompts using GPT-4o-mini
Usage: python generate_prompts.py
Requires: pip install openai
"""

import openai
import json
import os
import time
from datetime import datetime
import sys

def get_openai_client():
    """Initialize OpenAI client with API key"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("\nTo fix this:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Set it as an environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   # Or add it to your ~/.zshrc or ~/.bashrc")
        sys.exit(1)
    
    return openai.OpenAI(api_key=api_key)

def generate_question_batch(client, batch_num, questions_per_batch=10):
    """Generate a batch of common questions"""
    prompt = f"""You are GPT-4o-mini. Please generate {questions_per_batch} diverse questions that people commonly ask you across different categories. 

Make them:
- Varied in topic (science, philosophy, practical advice, creativity, etc.)
- Different complexity levels (simple to complex)
- Representative of real user queries
- Suitable as prompts for testing a language model

Format as a JSON list of strings. Example:
["What is the meaning of life?", "How do I cook pasta?", "Explain quantum physics"]

Generate {questions_per_batch} questions now:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse, commonly asked questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Higher temperature for more diversity
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            questions = json.loads(response_text)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
            else:
                print(f"⚠️  Batch {batch_num}: Response not a list of strings, trying to extract...")
                return extract_questions_fallback(response_text)
        except json.JSONDecodeError:
            print(f"⚠️  Batch {batch_num}: JSON parsing failed, trying to extract...")
            return extract_questions_fallback(response_text)
    
    except Exception as e:
        print(f"❌ Error in batch {batch_num}: {e}")
        return []

def extract_questions_fallback(text):
    """Fallback method to extract questions from non-JSON response"""
    lines = text.split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        # Look for quoted strings or numbered items
        if line.startswith('"') and line.endswith('"'):
            questions.append(line[1:-1])  # Remove quotes
        elif line.startswith('"') and line.endswith('",'):
            questions.append(line[1:-2])  # Remove quotes and comma
        elif any(line.startswith(f"{i}.") for i in range(1, 101)):
            # Handle numbered lists like "1. Question here"
            question = line.split('.', 1)[1].strip()
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            questions.append(question)
        elif line and '?' in line:
            # Any line with a question mark
            questions.append(line)
    
    return questions[:10]  # Limit to expected batch size

def save_prompts(all_questions, filename=None):
    """Save the generated prompts to files"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"common_prompts_{timestamp}"
    
    # Save as JSON
    json_file = f"{filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(all_questions),
            "questions": all_questions
        }, f, indent=2, ensure_ascii=False)
    
    # Save as plain text (one per line)
    txt_file = f"{filename}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"# Common Questions Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total: {len(all_questions)} questions\n\n")
        for i, question in enumerate(all_questions, 1):
            f.write(f"{i:3d}. {question}\n")
    
    return json_file, txt_file

def main():
    print("🤖 Generating Common Question Prompts using GPT-4o-mini")
    print("=" * 60)
    
    # Initialize OpenAI client
    client = get_openai_client()
    print("✅ OpenAI client initialized")
    
    # Configuration
    total_questions = 100
    questions_per_batch = 10
    num_batches = total_questions // questions_per_batch
    
    print(f"📋 Generating {total_questions} questions in {num_batches} batches of {questions_per_batch}")
    print("⏳ This may take a few minutes due to API rate limits...\n")
    
    all_questions = []
    
    for batch_num in range(1, num_batches + 1):
        print(f"🔄 Generating batch {batch_num}/{num_batches}...")
        
        questions = generate_question_batch(client, batch_num, questions_per_batch)
        
        if questions:
            all_questions.extend(questions)
            print(f"   ✅ Got {len(questions)} questions (Total: {len(all_questions)})")
            
            # Show a sample question from this batch
            if questions:
                print(f"   📝 Sample: \"{questions[0]}\"")
        else:
            print(f"   ❌ Failed to get questions for batch {batch_num}")
        
        # Rate limiting - be nice to the API
        if batch_num < num_batches:
            print("   ⏸️  Waiting 2 seconds...")
            time.sleep(2)
        
        print()
    
    # Remove duplicates while preserving order
    unique_questions = []
    seen = set()
    for q in all_questions:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            unique_questions.append(q)
            seen.add(q_lower)
    
    print(f"📊 Results Summary:")
    print(f"   - Total generated: {len(all_questions)}")
    print(f"   - Unique questions: {len(unique_questions)}")
    print(f"   - Duplicates removed: {len(all_questions) - len(unique_questions)}")
    
    if len(unique_questions) < 90:
        print(f"⚠️  Warning: Only got {len(unique_questions)} unique questions (target was {total_questions})")
    
    # Save to files
    print("\n💾 Saving prompts...")
    json_file, txt_file = save_prompts(unique_questions)
    
    print(f"✅ Saved to:")
    print(f"   - JSON format: {json_file}")
    print(f"   - Text format: {txt_file}")
    
    # Show some sample questions
    print(f"\n📋 Sample Questions Generated:")
    for i, question in enumerate(unique_questions[:5], 1):
        print(f"   {i}. {question}")
    
    if len(unique_questions) > 5:
        print(f"   ... and {len(unique_questions) - 5} more")
    
    print(f"\n🎉 Complete! You can now use these prompts to test your FactLM model:")
    print(f"   python generate_text.py models/your_model.pth")

if __name__ == "__main__":
    main() 