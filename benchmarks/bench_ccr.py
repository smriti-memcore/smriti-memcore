import time
import json
import urllib.request
from smriti_memcore.compressors.code_crusher import crush_code
from smriti_memcore.compressors.json_crusher import crush_json

def benchmark_code_compression():
    print("--- Code Compression Benchmark ---")
    
    # Read our own core.py to use as a realistic large code payload
    with open("smriti_memcore/core.py", "r") as f:
        code_text = f.read()
        
    start_time = time.perf_counter()
    compressed = crush_code(code_text)
    duration = (time.perf_counter() - start_time) * 1000  # ms
    
    orig_len = len(code_text)
    comp_len = len(compressed)
    ratio = comp_len / orig_len
    
    print(f"Original size:   {orig_len:,} chars")
    print(f"Compressed size: {comp_len:,} chars")
    print(f"Space saved:     {(1 - ratio):.1%}")
    print(f"Time taken:      {duration:.2f} ms")
    print()

def benchmark_json_compression():
    print("--- JSON Compression Benchmark ---")
    
    # Create a synthetic large JSON payload (e.g., typical list of dicts response)
    large_json = {
        "metadata": {
            "page": 1,
            "@context": "http://schema.org",
            "links": [{"rel": "next", "href": "https://api.example.com/v1/users?page=2"}],
            "total_count": 5000
        },
        "data": []
    }
    
    # Add 100 identical objects
    for i in range(100):
        large_json["data"].append({
            "id": f"user_{i}",
            "name": f"User {i}",
            "email": f"user{i}@example.com",
            "roles": ["admin", "editor"],
            "profile": {
                "avatar_url": "https://example.com/avatar.png",
                "bio": "This is a long bio " * 10
            }
        })
        
    json_text = json.dumps(large_json)
    
    start_time = time.perf_counter()
    compressed = crush_json(json_text)
    duration = (time.perf_counter() - start_time) * 1000  # ms
    
    orig_len = len(json_text)
    comp_len = len(compressed)
    ratio = comp_len / orig_len
    
    print(f"Original size:   {orig_len:,} chars")
    print(f"Compressed size: {comp_len:,} chars")
    print(f"Space saved:     {(1 - ratio):.1%}")
    print(f"Time taken:      {duration:.2f} ms")
    print()

if __name__ == "__main__":
    print("Starting CCR Benchmarks...\n")
    benchmark_code_compression()
    benchmark_json_compression()
