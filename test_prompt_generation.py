#!/usr/bin/env python3
"""
Test script to demonstrate prompt generation without making API calls

This script shows the variety of prompts that will be generated for GPT-4o-mini.
"""

from generate_training_data import PromptGenerator


def main():
    """Test prompt generation"""
    
    print("🧪 Testing FactLM Prompt Generation")
    print("=" * 40)
    
    generator = PromptGenerator()
    
    print(f"📊 Generator Info:")
    print(f"   Categories: {len(generator.categories)}")
    print(f"   Topics: {len(generator.topics)}")
    print(f"   Activities: {len(generator.activities)}")
    print(f"   Problems: {len(generator.problems)}")
    
    print(f"\n🔤 Categories:")
    for i, category in enumerate(generator.categories, 1):
        print(f"   {i:2d}. {category}")
    
    print(f"\n📝 Sample Prompts:")
    print("-" * 60)
    
    # Generate sample prompts from each category
    for category in generator.categories:
        print(f"\n🏷️  {category.upper().replace('_', ' ')}:")
        
        # Generate 2 prompts from this category by temporarily filtering
        samples = []
        attempts = 0
        while len(samples) < 2 and attempts < 20:
            prompt = generator.generate_prompt()
            # Simple heuristic to check if prompt matches category themes
            category_keywords = {
                'general_knowledge': ['what is', 'explain', 'facts about', 'how does'],
                'reasoning': ['conclude', 'logical', 'why might', 'factors'],
                'creative_writing': ['write', 'story', 'dialogue', 'poetic'],
                'problem_solving': ['solve', 'steps', 'approach', 'improve'],
                'explanation': ['explain', 'break down', 'teach', 'principles'],
                'analysis': ['pros and cons', 'analyze', 'patterns', 'causes'],
                'comparison': ['compare', 'contrast', 'similarities', 'differences'],
                'instruction_following': ['list', 'guide', 'instructions', 'how do you'],
                'conversation': ['say to someone', 'respond', 'advice', 'questions'],
                'factual_questions': ['facts about', 'when did', 'who was', 'where can']
            }
            
            keywords = category_keywords.get(category, [])
            if any(keyword in prompt.lower() for keyword in keywords):
                samples.append(prompt)
            attempts += 1
        
        # If we couldn't find category-specific prompts, just show random ones
        if not samples:
            samples = [generator.generate_prompt() for _ in range(2)]
        
        for i, prompt in enumerate(samples, 1):
            print(f"   {i}. {prompt}")
    
    print(f"\n" + "=" * 60)
    print(f"🎯 Random Sample (10 prompts):")
    print("-" * 60)
    
    for i in range(10):
        prompt = generator.generate_prompt()
        print(f"{i+1:2d}. {prompt}")
    
    print(f"\n✅ Prompt generation test complete!")
    print(f"   This shows the variety of prompts that will be sent to GPT-4o-mini")
    print(f"   To generate actual training data, run: python generate_training_data.py")


if __name__ == "__main__":
    main() 