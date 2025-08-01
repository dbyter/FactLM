"""
Training Data Generation Script

This script generates diverse prompts and gets responses from GPT-4o-mini
to create a custom training dataset for FactLM.
"""

import openai
import json
import time
import random
from typing import List, Dict, Any
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading


class PromptGenerator:
    """Generates diverse prompts for training data creation"""
    
    def __init__(self):
        self.categories = [
            "general_knowledge",
            "reasoning",
            "creative_writing",
            "problem_solving",
            "explanation",
            "analysis",
            "comparison",
            "instruction_following",
            "conversation",
            "factual_questions",
            "location_based"
        ]
        
        self.prompt_templates = {
            "general_knowledge": [
                "What is {topic} and why is it important?",
                "Explain the history of {topic}.",
                "What are the key facts about {topic}?",
                "How does {topic} work?",
                "What are the main types of {topic}?",
            ],
            "reasoning": [
                "If {premise}, then what can we conclude about {topic}?",
                "Why might someone think that {statement}? What evidence supports or contradicts this?",
                "What are the logical steps to solve this problem: {problem}?",
                "How would you approach analyzing {situation}?",
                "What factors should be considered when evaluating {topic}?",
            ],
            "creative_writing": [
                "Write a short story about {topic}.",
                "Create a dialogue between two people discussing {topic}.",
                "Describe {topic} in a poetic way.",
                "Write instructions for {activity} in an engaging style.",
                "Create a fictional scenario involving {topic}.",
            ],
            "problem_solving": [
                "How would you solve this problem: {problem}?",
                "What steps would you take to {goal}?",
                "If you encountered {challenge}, how would you handle it?",
                "What's the best approach to {task}?",
                "How can someone improve their {skill}?",
            ],
            "explanation": [
                "Explain {concept} to someone who has never heard of it before.",
                "Break down {process} into simple steps.",
                "What makes {topic} work the way it does?",
                "How would you teach {subject} to a beginner?",
                "What are the key principles behind {topic}?",
            ],
            "analysis": [
                "What are the pros and cons of {topic}?",
                "Analyze the impact of {event} on {domain}.",
                "What patterns can you identify in {data_type}?",
                "How has {topic} changed over time?",
                "What are the underlying causes of {phenomenon}?",
            ],
            "comparison": [
                "Compare and contrast {item1} and {item2}.",
                "What are the similarities and differences between {concept1} and {concept2}?",
                "Which is better: {option1} or {option2}? Why?",
                "How does {topic} in {context1} differ from {topic} in {context2}?",
                "What are the relative advantages of {approach1} vs {approach2}?",
            ],
            "instruction_following": [
                "Please list the steps to {task}.",
                "Create a guide for {activity}.",
                "What should someone do if they want to {goal}?",
                "Give me instructions for {process}.",
                "How do you {action} properly?",
            ],
            "conversation": [
                "What would you say to someone who believes {controversial_statement}?",
                "How would you respond to someone asking about {topic}?",
                "What advice would you give to someone struggling with {challenge}?",
                "How would you explain {complex_topic} in a conversation?",
                "What questions would you ask to learn more about {subject}?",
            ],
            "factual_questions": [
                "What are some interesting facts about {topic}?",
                "When did {event} happen and why was it significant?",
                "Who was {person} and what did they contribute to {field}?",
                "Where can you find {item} and what is it used for?",
                "What is the current status of {ongoing_topic}?",
                "What makes {topic} unique compared to other places?",
                "How has {topic} changed over the past century?",
                "What cultural significance does {topic} hold?",
                "What are the main attractions in {topic}?",
                "What is the history behind {topic}?",
            ],
            "location_based": [
                "What would you recommend for someone visiting {topic} for the first time?",
                "How has {topic} influenced global culture and history?",
                "What are the best times of year to visit {topic}?",
                "What local customs and traditions are important to know about {topic}?",
                "How has {topic} evolved from its origins to the present day?",
                "What makes {topic} a significant destination for travelers?",
                "What are the environmental challenges facing {topic}?",
                "How does {topic} compare to similar places around the world?",
                "What economic factors have shaped the development of {topic}?",
                "What role does {topic} play in its region or country?",
            ]
        }
        
        # Topics and entities to fill in templates
        self.topics = [
            "artificial intelligence", "climate change", "renewable energy", "space exploration",
            "quantum computing", "biotechnology", "cryptocurrency", "machine learning",
            "sustainable development", "digital privacy", "robotics", "virtual reality",
            "genetic engineering", "solar power", "electric vehicles", "3D printing",
            "blockchain technology", "cybersecurity", "automation", "data science",
            "neuroscience", "psychology", "philosophy", "literature", "art history",
            "music theory", "economics", "politics", "sociology", "anthropology",
            "mathematics", "physics", "chemistry", "biology", "geology",
            "programming", "web development", "mobile apps", "cloud computing",
            "fitness", "nutrition", "meditation", "cooking", "gardening",
            "travel", "photography", "languages", "education", "healthcare",
            # Countries and regions
            "United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Peru", "Colombia",
            "United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands", "Belgium", "Switzerland",
            "Russia", "China", "Japan", "South Korea", "India", "Pakistan", "Bangladesh", "Thailand",
            "Vietnam", "Indonesia", "Malaysia", "Singapore", "Philippines", "Australia", "New Zealand",
            "South Africa", "Nigeria", "Kenya", "Egypt", "Morocco", "Tunisia", "Algeria", "Ethiopia",
            "Saudi Arabia", "Iran", "Iraq", "Turkey", "Israel", "Jordan", "Lebanon", "Syria",
            "Ukraine", "Poland", "Czech Republic", "Hungary", "Romania", "Bulgaria", "Greece", "Portugal",
            "Norway", "Sweden", "Finland", "Denmark", "Iceland", "Ireland", "Austria", "Slovenia",
            # Major cities
            "New York City", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
            "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus",
            "London", "Paris", "Berlin", "Madrid", "Rome", "Amsterdam", "Brussels", "Vienna",
            "Moscow", "Saint Petersburg", "Beijing", "Shanghai", "Tokyo", "Seoul", "Delhi", "Mumbai",
            "Bangkok", "Jakarta", "Manila", "Singapore", "Sydney", "Melbourne", "Auckland",
            "Cairo", "Lagos", "Nairobi", "Johannesburg", "Cape Town", "Casablanca", "Algiers",
            "Riyadh", "Tehran", "Baghdad", "Istanbul", "Jerusalem", "Amman", "Beirut",
            "Kiev", "Warsaw", "Prague", "Budapest", "Bucharest", "Sofia", "Athens", "Lisbon",
            "Oslo", "Stockholm", "Helsinki", "Copenhagen", "Reykjavik", "Dublin", "Ljubljana",
            # Landmarks and tourist destinations
            "Eiffel Tower", "Statue of Liberty", "Big Ben", "Tower Bridge", "Colosseum", "Vatican City",
            "Sagrada Familia", "Acropolis", "Parthenon", "Taj Mahal", "Great Wall of China", "Forbidden City",
            "Mount Fuji", "Petra", "Pyramids of Giza", "Sphinx", "Stonehenge", "Machu Picchu",
            "Christ the Redeemer", "Sydney Opera House", "Golden Gate Bridge", "Empire State Building",
            "Times Square", "Central Park", "Yellowstone National Park", "Grand Canyon", "Yosemite",
            "Niagara Falls", "Disney World", "Disneyland", "Universal Studios", "Hollywood",
            "Las Vegas Strip", "Miami Beach", "Waikiki Beach", "Mount Rushmore", "White House",
            "Buckingham Palace", "Palace of Versailles", "Neuschwanstein Castle", "Alhambra",
            "Mont Saint-Michel", "Pompeii", "Herculaneum", "Pompeii", "Herculaneum",
            # Natural features and geography
            "Amazon Rainforest", "Sahara Desert", "Gobi Desert", "Arabian Desert", "Kalahari Desert",
            "Rocky Mountains", "Appalachian Mountains", "Alps", "Himalayas", "Andes", "Atlas Mountains",
            "Mississippi River", "Nile River", "Amazon River", "Yangtze River", "Yellow River",
            "Danube River", "Rhine River", "Thames River", "Seine River", "Volga River",
            "Pacific Ocean", "Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Mediterranean Sea",
            "Caribbean Sea", "Red Sea", "Black Sea", "Caspian Sea", "Dead Sea",
            "Mount Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu", "Cho Oyu",
            "Mount Kilimanjaro", "Mount Kenya", "Mount Elbrus", "Mont Blanc", "Matterhorn",
            # Historical and cultural sites
            "Angkor Wat", "Borobudur", "Prambanan", "Shwedagon Pagoda", "Temple of the Emerald Buddha",
            "Senso-ji", "Meiji Shrine", "Fushimi Inari Shrine", "Kinkaku-ji", "Ginkaku-ji",
            "Hagia Sophia", "Blue Mosque", "Topkapi Palace", "Dolmabahce Palace", "Cappadocia",
            "Pamukkale", "Ephesus", "Troy", "Gordion", "Hattusa", "Gobekli Tepe",
            "Persepolis", "Pasargadae", "Susa", "Ecbatana", "Ctesiphon", "Babylon",
            "Ur", "Uruk", "Nineveh", "Assur", "Nimrud", "Hatra", "Palmyra", "Dura-Europos",
            "Petra", "Jerash", "Umm Qais", "Ajloun Castle", "Karak Castle", "Shobak Castle",
            "Wadi Rum", "Dead Sea", "Jordan River", "Mount Nebo", "Madaba", "Mount of Olives",
            "Church of the Holy Sepulchre", "Western Wall", "Dome of the Rock", "Al-Aqsa Mosque",
            "Bethlehem", "Nazareth", "Sea of Galilee", "Mount Tabor", "Mount Carmel",
            "Haifa", "Tel Aviv", "Jaffa", "Caesarea", "Acre", "Safed", "Tiberias",
            "Jericho", "Hebron", "Nablus", "Ramallah", "Jenin", "Tulkarm", "Qalqilya",
            "Salfit", "Tubas", "Bethlehem", "Hebron", "Jericho", "Jenin", "Nablus",
            "Ramallah", "Tulkarm", "Qalqilya", "Salfit", "Tubas", "Tubas", "Tubas",
            # Modern cities and urban areas
            "Silicon Valley", "Wall Street", "Hollywood", "Broadway", "Times Square",
            "Chinatown", "Little Italy", "Harlem", "Brooklyn", "Queens", "Bronx", "Staten Island",
            "Manhattan", "Downtown", "Uptown", "Midtown", "Upper East Side", "Upper West Side",
            "Lower East Side", "Lower West Side", "Chelsea", "Greenwich Village", "SoHo", "NoHo",
            "Tribeca", "Financial District", "Battery Park", "Battery Park City", "World Trade Center",
            "Ground Zero", "9/11 Memorial", "One World Trade Center", "Freedom Tower",
            "Empire State Building", "Chrysler Building", "Rockefeller Center", "Radio City Music Hall",
            "Carnegie Hall", "Lincoln Center", "Metropolitan Museum of Art", "Museum of Modern Art",
            "American Museum of Natural History", "Guggenheim Museum", "Whitney Museum",
            "Brooklyn Museum", "Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge",
            "Queensboro Bridge", "George Washington Bridge", "Verrazzano-Narrows Bridge",
            "Holland Tunnel", "Lincoln Tunnel", "Queens-Midtown Expressway", "Brooklyn-Queens Expressway",
            "FDR Drive", "West Side Highway", "East River", "Hudson River", "Harlem River",
            "Long Island Sound", "Jamaica Bay", "New York Harbor", "Upper New York Bay",
            "Lower New York Bay", "The Narrows", "Staten Island Sound", "Arthur Kill",
            "Kill Van Kull", "Newark Bay", "Raritan Bay", "Sandy Hook Bay", "Jamaica Bay",
            "Flushing Bay", "Little Neck Bay", "Manhasset Bay", "Hempstead Harbor",
            "Oyster Bay", "Cold Spring Harbor", "Huntington Harbor", "Northport Harbor",
            "Port Jefferson Harbor", "Mount Sinai Harbor", "Stony Brook Harbor", "Setauket Harbor",
            "Conscience Bay", "Nissequogue River", "Connetquot River", "Carmans River",
            "Peconic River", "Carmans River", "Connetquot River", "Nissequogue River",
            "Conscience Bay", "Setauket Harbor", "Stony Brook Harbor", "Mount Sinai Harbor",
            "Port Jefferson Harbor", "Northport Harbor", "Huntington Harbor", "Cold Spring Harbor",
            "Oyster Bay", "Hempstead Harbor", "Manhasset Bay", "Little Neck Bay", "Flushing Bay",
            "Jamaica Bay", "Sandy Hook Bay", "Raritan Bay", "Newark Bay", "Kill Van Kull",
            "Arthur Kill", "Staten Island Sound", "The Narrows", "Lower New York Bay",
            "Upper New York Bay", "New York Harbor", "Jamaica Bay", "Long Island Sound",
            "Harlem River", "Hudson River", "East River", "West Side Highway", "FDR Drive",
            "Brooklyn-Queens Expressway", "Queens-Midtown Expressway", "Lincoln Tunnel",
            "Holland Tunnel", "Verrazzano-Narrows Bridge", "George Washington Bridge",
            "Queensboro Bridge", "Williamsburg Bridge", "Manhattan Bridge", "Brooklyn Bridge",
            "Whitney Museum", "Guggenheim Museum", "American Museum of Natural History",
            "Museum of Modern Art", "Metropolitan Museum of Art", "Lincoln Center",
            "Carnegie Hall", "Radio City Music Hall", "Rockefeller Center", "Chrysler Building",
            "Empire State Building", "Freedom Tower", "One World Trade Center", "9/11 Memorial",
            "Ground Zero", "World Trade Center", "Battery Park City", "Battery Park",
            "Financial District", "Tribeca", "NoHo", "SoHo", "Greenwich Village", "Chelsea",
            "Upper West Side", "Upper East Side", "Midtown", "Uptown", "Downtown", "Manhattan",
            "Staten Island", "Bronx", "Queens", "Brooklyn", "Harlem", "Little Italy", "Chinatown",
            "Broadway", "Hollywood", "Wall Street", "Silicon Valley", "Tubas", "Tubas", "Tubas",
            "Tubas", "Salfit", "Qalqilya", "Tulkarm", "Nablus", "Jenin", "Jericho", "Hebron",
            "Bethlehem", "Tubas", "Salfit", "Qalqilya", "Tulkarm", "Nablus", "Jenin", "Jericho",
            "Hebron", "Bethlehem", "Tiberias", "Safed", "Acre", "Caesarea", "Jaffa", "Tel Aviv",
            "Haifa", "Mount Carmel", "Mount Tabor", "Sea of Galilee", "Nazareth", "Bethlehem",
            "Al-Aqsa Mosque", "Dome of the Rock", "Western Wall", "Church of the Holy Sepulchre",
            "Mount of Olives", "Madaba", "Mount Nebo", "Jordan River", "Dead Sea", "Wadi Rum",
            "Shobak Castle", "Karak Castle", "Ajloun Castle", "Umm Qais", "Jerash", "Petra",
            "Dura-Europos", "Palmyra", "Hatra", "Nimrud", "Assur", "Nineveh", "Uruk", "Ur",
            "Ctesiphon", "Ecbatana", "Susa", "Pasargadae", "Persepolis", "Gobekli Tepe", "Hattusa",
            "Gordion", "Troy", "Ephesus", "Pamukkale", "Cappadocia", "Dolmabahce Palace",
            "Topkapi Palace", "Blue Mosque", "Hagia Sophia", "Ginkaku-ji", "Kinkaku-ji",
            "Fushimi Inari Shrine", "Meiji Shrine", "Senso-ji", "Temple of the Emerald Buddha",
            "Shwedagon Pagoda", "Prambanan", "Borobudur", "Angkor Wat", "Matterhorn", "Mont Blanc",
            "Mount Elbrus", "Mount Kenya", "Mount Kilimanjaro", "Cho Oyu", "Makalu", "Lhotse",
            "Kangchenjunga", "K2", "Mount Everest", "Dead Sea", "Caspian Sea", "Black Sea",
            "Red Sea", "Caribbean Sea", "Mediterranean Sea", "Arctic Ocean", "Indian Ocean",
            "Atlantic Ocean", "Pacific Ocean", "Volga River", "Seine River", "Thames River",
            "Rhine River", "Danube River", "Yellow River", "Yangtze River", "Amazon River",
            "Nile River", "Mississippi River", "Atlas Mountains", "Andes", "Himalayas", "Alps",
            "Appalachian Mountains", "Rocky Mountains", "Kalahari Desert", "Arabian Desert",
            "Gobi Desert", "Sahara Desert", "Amazon Rainforest", "White House", "Mount Rushmore",
            "Waikiki Beach", "Miami Beach", "Las Vegas Strip", "Hollywood", "Universal Studios",
            "Disneyland", "Disney World", "Niagara Falls", "Yosemite", "Grand Canyon",
            "Yellowstone National Park", "Central Park", "Times Square", "Empire State Building",
            "Golden Gate Bridge", "Sydney Opera House", "Christ the Redeemer", "Machu Picchu",
            "Stonehenge", "Sphinx", "Pyramids of Giza", "Petra", "Mount Fuji", "Forbidden City",
            "Great Wall of China", "Taj Mahal", "Parthenon", "Acropolis", "Vatican City",
            "Colosseum", "Tower Bridge", "Big Ben", "Statue of Liberty", "Eiffel Tower",
            "Ljubljana", "Dublin", "Reykjavik", "Copenhagen", "Helsinki", "Stockholm", "Oslo",
            "Lisbon", "Athens", "Sofia", "Bucharest", "Budapest", "Prague", "Warsaw", "Kiev",
            "Beirut", "Amman", "Jerusalem", "Istanbul", "Baghdad", "Tehran", "Riyadh", "Algiers",
            "Casablanca", "Cape Town", "Johannesburg", "Nairobi", "Lagos", "Cairo", "Auckland",
            "Melbourne", "Sydney", "Singapore", "Manila", "Jakarta", "Bangkok", "Mumbai", "Delhi",
            "Seoul", "Tokyo", "Shanghai", "Beijing", "Saint Petersburg", "Moscow", "Vienna",
            "Brussels", "Amsterdam", "Rome", "Madrid", "Berlin", "Paris", "London", "Columbus",
            "Fort Worth", "Jacksonville", "Austin", "San Jose", "Dallas", "San Diego", "San Antonio",
            "Philadelphia", "Phoenix", "Houston", "Chicago", "Los Angeles", "New York City",
            "Slovenia", "Austria", "Ireland", "Iceland", "Denmark", "Finland", "Sweden", "Norway",
            "Portugal", "Greece", "Bulgaria", "Romania", "Hungary", "Czech Republic", "Poland",
            "Ukraine", "Syria", "Lebanon", "Jordan", "Israel", "Turkey", "Iraq", "Iran",
            "Saudi Arabia", "Ethiopia", "Algeria", "Tunisia", "Morocco", "Egypt", "Kenya",
            "Nigeria", "South Africa", "New Zealand", "Australia", "Philippines", "Singapore",
            "Malaysia", "Indonesia", "Vietnam", "Thailand", "Bangladesh", "Pakistan", "India",
            "South Korea", "Japan", "China", "Russia", "Switzerland", "Belgium", "Netherlands",
            "Spain", "Italy", "Germany", "France", "United Kingdom", "Staten Island", "Bronx",
            "Queens", "Brooklyn", "Manhattan", "Chile", "Peru", "Colombia", "Argentina", "Brazil",
            "Mexico", "Canada", "United States"
        ]
        
        self.activities = [
            "learning a new language", "starting a business", "writing a book",
            "building a website", "growing vegetables", "training for a marathon",
            "playing an instrument", "painting a portrait", "debugging code",
            "planning a trip", "organizing files", "managing time", "studying effectively"
        ]
        
        self.problems = [
            "a team member not meeting deadlines", "low website traffic",
            "difficulty concentrating", "budget constraints", "technical errors",
            "communication breakdown", "creative block", "time management issues",
            "conflicting priorities", "lack of motivation", "information overload"
        ]

    def generate_prompt(self) -> str:
        """Generate a single diverse prompt"""
        category = random.choice(self.categories)
        template = random.choice(self.prompt_templates[category])
        
        # Fill in template placeholders
        prompt = template
        
        # Replace common placeholders
        replacements = {
            "{topic}": random.choice(self.topics),
            "{activity}": random.choice(self.activities),
            "{problem}": random.choice(self.problems),
            "{concept}": random.choice(self.topics),
            "{process}": random.choice(self.activities),
            "{subject}": random.choice(self.topics),
            "{task}": random.choice(self.activities),
            "{goal}": "improve " + random.choice(self.topics),
            "{challenge}": random.choice(self.problems),
            "{skill}": random.choice(self.topics),
            "{event}": "the development of " + random.choice(self.topics),
            "{domain}": random.choice(self.topics),
            "{data_type}": random.choice(["user behavior", "market trends", "research data"]),
            "{phenomenon}": random.choice(self.topics),
            "{item1}": random.choice(self.topics),
            "{item2}": random.choice(self.topics),
            "{concept1}": random.choice(self.topics),
            "{concept2}": random.choice(self.topics),
            "{option1}": random.choice(self.topics),
            "{option2}": random.choice(self.topics),
            "{context1}": random.choice(["education", "business", "healthcare", "technology", "government", "non-profit"]),
            "{context2}": random.choice(["business", "education", "healthcare", "technology", "government", "non-profit"]),
            "{approach1}": random.choice(["traditional methods", "modern techniques", "digital approaches", "analog methods", "conventional practices", "innovative strategies"]),
            "{approach2}": random.choice(["modern techniques", "traditional methods", "digital approaches", "analog methods", "conventional practices", "innovative strategies"]),
            "{controversial_statement}": random.choice([
                "technology is making people less social",
                "social media is harmful to society",
                "artificial intelligence will replace human workers",
                "climate change is not a serious threat",
                "vaccines are dangerous",
                "the government should control the internet",
                "privacy is more important than security",
                "traditional education is outdated"
            ]),
            "{complex_topic}": random.choice(self.topics),
            "{ongoing_topic}": random.choice(self.topics),
            "{person}": random.choice([
                "a famous scientist", "a renowned artist", "a historical figure", 
                "a political leader", "a business innovator", "a cultural icon",
                "a sports legend", "a literary figure", "a musical genius", "a philosopher"
            ]),
            "{field}": random.choice(self.topics),
            "{item}": random.choice(self.topics),
            "{statement}": random.choice(self.topics) + " is overrated",
            "{premise}": random.choice([
                "technology continues to advance rapidly",
                "globalization is increasing",
                "climate change is accelerating",
                "urbanization is growing worldwide",
                "digital transformation is reshaping industries",
                "population growth is slowing in developed countries",
                "renewable energy adoption is increasing",
                "automation is replacing manual labor"
            ]),
            "{situation}": random.choice([
                "a new market opportunity", "a technological breakthrough", 
                "a natural disaster", "an economic crisis", "a political change",
                "a social movement", "an environmental challenge", "a cultural shift"
            ]),
            "{action}": "implement " + random.choice(self.topics)
        }
        
        for placeholder, replacement in replacements.items():
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, replacement)
        
        return prompt
    
    def generate_batch(self, batch_size: int) -> List[str]:
        """Generate a batch of diverse prompts"""
        return [self.generate_prompt() for _ in range(batch_size)]


class TrainingDataGenerator:
    """Main class for generating training data using GPT-4o-mini"""
    
    def __init__(self, api_key: str = None, max_concurrent_requests: int = 50, request_timeout: float = 30.0):
        """Initialize with OpenAI API key, concurrency settings, and timeout"""
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.prompt_generator = PromptGenerator()
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.request_lock = threading.Lock()
        self.request_count = 0
        self.successful_count = 0
        self.failed_count = 0
    
    def get_single_response(self, prompt: str, prompt_index: int) -> Dict[str, Any]:
        """Get response from GPT-4o-mini for a single prompt with function-level timeout"""
        
        # Pre-assign request number for consistent error reporting
        with self.request_lock:
            self.request_count += 1
            current_request_num = self.request_count
        
        def _make_api_call():
            """Internal function to make the actual API call"""            
            print(f"  🔄 Request {current_request_num}: {prompt[:50]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful, knowledgeable assistant. Provide clear, accurate, and informative responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
                # Removed unreliable OpenAI timeout parameter
            )
            
            answer = response.choices[0].message.content
            
            with self.request_lock:
                self.successful_count += 1
            
            result = {
                "prompt": prompt,
                "response": answer,
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o-mini",
                "success": True,
                "request_index": prompt_index
            }
            
            print(f"  ✅ Request {current_request_num}: Success ({len(answer)} chars)")
            return result
        
        # Use ThreadPoolExecutor to implement reliable function-level timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_make_api_call)
                result = future.result(timeout=self.request_timeout)
                return result
                
        except Exception as e:
            with self.request_lock:
                self.failed_count += 1
            
            # Check if it's a timeout error for better error reporting
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower() or isinstance(e, TimeoutError):
                print(f"  ⏰ Request {current_request_num}: Function timeout after {self.request_timeout} seconds")
            else:
                print(f"  ❌ Request {current_request_num}: Error - {e}")
            
            return {
                "prompt": prompt,
                "response": None,
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o-mini",
                "success": False,
                "error": str(e),
                "request_index": prompt_index
            }
    
    def get_response_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Get responses from GPT-4o-mini for a batch of prompts using concurrent processing"""
        print(f"  🚀 Processing {len(prompts)} prompts concurrently (max {self.max_concurrent_requests} at once)")
        
        # Reset counters for this batch
        batch_start_time = time.time()
        
        results = [None] * len(prompts)  # Pre-allocate results list to maintain order
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all prompts as futures
            future_to_index = {
                executor.submit(self.get_single_response, prompt, i): i 
                for i, prompt in enumerate(prompts)
            }
            
            # Process completed futures as they finish
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                results[index] = result
                
                completed_count += 1
                
                # Progress update every 10 completions or at the end
                if completed_count % 10 == 0 or completed_count == len(prompts):
                    elapsed = time.time() - batch_start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    print(f"    📊 Progress: {completed_count}/{len(prompts)} complete ({rate:.1f} req/sec)")
        
        # Final batch statistics
        batch_time = time.time() - batch_start_time
        successful_in_batch = sum(1 for r in results if r and r['success'])
        failed_in_batch = len(results) - successful_in_batch
        
        print(f"  ✅ Batch complete: {successful_in_batch} success, {failed_in_batch} failed in {batch_time:.1f}s")
        
        # Add a small delay between batches to be respectful to the API
        time.sleep(1.0)
        
        return results
    
    def save_training_data(self, results: List[Dict[str, Any]], output_file: str):
        """Save training data in a format compatible with data_loader.py"""
        
        # Filter successful results
        successful_results = [r for r in results if r and r['success'] and r['response']]
        
        print(f"Saving {len(successful_results)} successful conversations to {output_file}")
        
        # Format as conversations for the data loader
        formatted_conversations = []
        for result in successful_results:
            conversation = {
                "data": [
                    result["prompt"],  # Human message
                    result["response"]  # Assistant response
                ],
                "metadata": {
                    "timestamp": result["timestamp"],
                    "model": result["model"],
                    "source": "generated_training_data"
                }
            }
            formatted_conversations.append(conversation)
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_conversations, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(formatted_conversations)} conversations to {output_file}")
        
        # Also save raw results for debugging
        debug_file = output_file.replace('.json', '_raw.json')
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved raw results to {debug_file}")
    
    def generate_training_dataset(self, 
                                  total_prompts: int = 10000, 
                                  batch_size: int = 100,
                                  output_file: str = "generated_training_data.json"):
        """Generate the complete training dataset using concurrent processing"""
        
        print(f"🚀 Starting concurrent training data generation:")
        print(f"   Target: {total_prompts:,} prompts")
        print(f"   Batch size: {batch_size}")
        print(f"   Max concurrent requests: {self.max_concurrent_requests}")
        print(f"   Output file: {output_file}")
        
        all_results = []
        num_batches = (total_prompts + batch_size - 1) // batch_size
        overall_start_time = time.time()
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_prompts)
            current_batch_size = end_idx - start_idx
            
            print(f"\n📦 Processing batch {batch_num + 1}/{num_batches} ({current_batch_size} prompts)")
            
            # Generate prompts for this batch
            prompts = self.prompt_generator.generate_batch(current_batch_size)
            
            # Get responses concurrently
            batch_results = self.get_response_batch(prompts)
            all_results.extend(batch_results)
            
            # Progress update
            successful_in_batch = sum(1 for r in batch_results if r and r['success'])
            total_successful = sum(1 for r in all_results if r and r['success'])
            total_failed = len(all_results) - total_successful
            
            elapsed_time = time.time() - overall_start_time
            avg_rate = len(all_results) / elapsed_time if elapsed_time > 0 else 0
            
            print(f"   ✅ Batch {batch_num + 1} complete: {successful_in_batch}/{current_batch_size} successful")
            print(f"   📊 Overall progress: {len(all_results)}/{total_prompts} prompts, {total_successful} successful, {total_failed} failed")
            print(f"   ⚡ Average rate: {avg_rate:.1f} prompts/sec")
            
            # Save intermediate results every 5 batches
            if (batch_num + 1) % 5 == 0:
                temp_file = f"temp_{output_file}"
                self.save_training_data(all_results, temp_file) 
                print(f"   💾 Intermediate save to {temp_file}")
        
        # Save final results
        print(f"\n💾 Saving final results...")
        self.save_training_data(all_results, output_file)
        
        # Final summary
        total_time = time.time() - overall_start_time
        successful_count = sum(1 for r in all_results if r and r['success'])
        overall_rate = len(all_results) / total_time if total_time > 0 else 0
        
        print(f"\n🎉 Training data generation complete!")
        print(f"   Total prompts processed: {len(all_results):,}")
        print(f"   Successful responses: {successful_count:,}")
        print(f"   Failed responses: {len(all_results) - successful_count:,}")
        print(f"   Success rate: {successful_count/len(all_results)*100:.1f}%")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Average rate: {overall_rate:.1f} prompts/sec")
        print(f"   Output file: {output_file}")
        
        return all_results


def main():
    """Main function to run the training data generation"""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Configuration
    TOTAL_PROMPTS = 10000
    BATCH_SIZE = 100
    MAX_CONCURRENT = 100  # Adjust based on your API rate limits
    REQUEST_TIMEOUT = 30.0  # 30-second timeout for each request
    OUTPUT_FILE = "generated_training_data.json"
    
    print(f"🔧 Configuration:")
    print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"   Concurrent requests: {MAX_CONCURRENT}")
    print(f"   Request timeout: {REQUEST_TIMEOUT} seconds")
    
    # Create generator with concurrency and timeout settings
    generator = TrainingDataGenerator(
        api_key=api_key,
        max_concurrent_requests=MAX_CONCURRENT,
        request_timeout=REQUEST_TIMEOUT
    )
    
    # Run generation
    generator.generate_training_dataset(
        total_prompts=TOTAL_PROMPTS,
        batch_size=BATCH_SIZE,
        output_file=OUTPUT_FILE
    )


if __name__ == "__main__":
    main() 