#!/usr/bin/env python3
"""
Test script for multithreaded Wikipedia processing in data_loader.py

This script demonstrates the performance benefits of multithreading for Wikipedia article processing
by comparing different numbers of worker threads.
"""

import time
import os
from data_loader import load_and_process_all_data

def test_multithreaded_processing():
    """Test Wikipedia processing with different numbers of threads"""
    
    print("🔬 Testing Multithreaded Wikipedia Processing")
    print("=" * 60)
    
    # Test configurations - using larger samples to better reflect 1M article processing
    test_configs = [
        {"workers": 1, "samples": 10000, "description": "Single-threaded baseline"},
        {"workers": 2, "samples": 10000, "description": "2 threads"},
        {"workers": 4, "samples": 10000, "description": "4 threads (default)"},
        {"workers": 8, "samples": 10000, "description": "8 threads"},
        {"workers": 12, "samples": 10000, "description": "12 threads (high-end)"},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['description']}")
        print(f"   Workers: {config['workers']}, Samples: {config['samples']:,}")
        
        start_time = time.time()
        
        try:
            train_data, val_data, stats = load_and_process_all_data(
                data_dir='data',
                ultrachat_samples=0,  # Disable UltraChat for faster testing
                wikipedia_samples=config['samples'],
                generated_data_file=None,  # Disable generated data
                max_workers=config['workers']
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            result = {
                'workers': config['workers'],
                'samples': config['samples'],
                'duration': duration,
                'total_tokens': stats['total_tokens'],
                'wikipedia_tokens': stats['wikipedia_tokens'],
                'description': config['description']
            }
            
            results.append(result)
            
            print(f"   ⏱️  Duration: {duration:.2f} seconds")
            print(f"   📊 Wikipedia tokens: {stats['wikipedia_tokens']:,}")
            print(f"   📈 Processing rate: {config['samples'] / duration:.1f} articles/second")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Print comparison results
    print(f"\n📊 Performance Comparison Results")
    print("=" * 60)
    
    if results:
        baseline = results[0]  # Single-threaded baseline
        
        print(f"{'Workers':<8} {'Duration':<10} {'Rate (art/s)':<12} {'Speedup':<8} {'Description'}")
        print("-" * 55)
        
        for result in results:
            speedup = baseline['duration'] / result['duration'] if result['duration'] > 0 else 0
            rate = result['samples'] / result['duration'] if result['duration'] > 0 else 0
            
            print(f"{result['workers']:<8} {result['duration']:<10.2f} {rate:<12.1f} {speedup:<8.2f}x {result['description']}")
        
        # Find optimal configuration
        if len(results) > 1:
            fastest = min(results, key=lambda x: x['duration'])
            print(f"\n🏆 Fastest configuration: {fastest['workers']} workers ({fastest['duration']:.2f}s)")
            print(f"   💡 Speedup vs single-thread: {baseline['duration'] / fastest['duration']:.2f}x")
            
            # Recommendations
            print(f"\n💡 Recommendations:")
            if fastest['workers'] <= 4:
                print(f"   - Use {fastest['workers']} workers for optimal performance")
            else:
                print(f"   - Your system benefits from higher thread counts")
                print(f"   - Consider using {fastest['workers']} workers for maximum speed")
            
            print(f"   - CPU cores detected: {os.cpu_count()}")
            print(f"   - For production use, start with max_workers={min(4, os.cpu_count() or 4)}")


def test_memory_usage():
    """Test memory usage with different worker configurations"""
    
    print(f"\n🧠 Memory Usage Test")
    print("=" * 60)
    
    try:
        import psutil
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Test with 4 workers
        print(f"\nTesting with 4 workers...")
        start_memory = process.memory_info().rss / 1024 / 1024
        
        train_data, val_data, stats = load_and_process_all_data(
            data_dir='data',
            ultrachat_samples=0,
            wikipedia_samples=1000,  # Smaller sample for memory test
            generated_data_file=None,
            max_workers=4
        )
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - start_memory
        
        print(f"Peak memory usage: {peak_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Memory per article: {memory_increase / 1000:.3f} MB")
        
    except ImportError:
        print("psutil not available - install with: pip install psutil")
    except Exception as e:
        print(f"Memory test failed: {e}")


if __name__ == "__main__":
    # Only run if data directory exists
    if not os.path.exists('data'):
        print("❌ Data directory not found. Please ensure you have book*.txt files in the 'data' directory.")
        exit(1)
    
    print("🚀 FactLM Multithreaded Processing Test")
    print("This script will test Wikipedia processing performance with different thread counts.")
    print("Note: This will download Wikipedia data if not already cached.")
    print("📊 Current default: 1M Wikipedia articles (estimated 10-30 minutes to process)")
    print("🧪 Test uses 10K articles per configuration for faster benchmarking")
    
    # Ask user if they want to proceed
    response = input("\nProceed with performance test? (y/N): ").strip().lower()
    if response != 'y':
        print("Test cancelled.")
        exit(0)
    
    try:
        test_multithreaded_processing()
        test_memory_usage()
        
        print(f"\n✅ All tests completed!")
        print(f"💡 Use the optimal worker count in your training scripts for best performance.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}") 