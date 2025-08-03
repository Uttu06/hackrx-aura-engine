"""
Rigorous latency testing script for FastAPI RAG endpoint.
Tests both cache miss and cache hit scenarios with detailed timing analysis.
Upgraded with Bearer token authentication and official test payload.
"""
import os
import requests
import time
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_URL = "http://127.0.0.1:8000/hackrx/run"

# Official test payload with new document and questions
payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}


def make_api_request(payload: Dict[str, Any], timeout: int = 180) -> tuple[requests.Response, float]:
    """
    Make API request with Bearer token authentication and measure response time.
    
    Args:
        payload: JSON payload to send
        timeout: Request timeout in seconds (increased for re-ranking pipeline)
        
    Returns:
        Tuple of (response object, elapsed time in seconds)
    """
    # Load Bearer token from environment
    API_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")
    if not API_TOKEN:
        raise ValueError("HACKRX_BEARER_TOKEN environment variable is required")
    
    # Prepare headers with authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=timeout,
            headers=headers
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        return response, elapsed_time
        
    except requests.exceptions.Timeout:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"❌ Request timed out after {elapsed_time:.2f} seconds")
        raise
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - make sure the FastAPI server is running on http://127.0.0.1:8000")
        raise
    except Exception as e:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"❌ Unexpected error after {elapsed_time:.2f} seconds: {e}")
        raise


def print_response_details(response: requests.Response, elapsed_time: float, run_type: str):
    """
    Print detailed response information.
    
    Args:
        response: HTTP response object
        elapsed_time: Time taken for the request
        run_type: Description of the test run (e.g., "Cache Miss", "Cache Hit")
    """
    print(f"📊 {run_type} Results:")
    print(f"   HTTP Status Code: {response.status_code}")
    print(f"   Response Time: {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        try:
            response_data = response.json()
            answers = response_data.get("answers", [])
            print(f"   Number of Answers: {len(answers)}")
            print(f"   ✅ Request successful")
            
            # Show first answer as sample
            if answers:
                first_answer = answers[0][:100] + "..." if len(answers[0]) > 100 else answers[0]
                print(f"   Sample Answer: {first_answer}")
                
        except json.JSONDecodeError:
            print(f"   ⚠️  Response is not valid JSON")
            print(f"   Response preview: {response.text[:200]}...")
    else:
        print(f"   ❌ Request failed")
        print(f"   Error message: {response.text}")
    
    print()


def test_api_latency():
    """
    Main function to test API latency for cache miss and cache hit scenarios.
    """
    print("🚀 Starting RAG API Latency Test")
    print(f"📍 Target URL: {API_URL}")
    print(f"📝 Testing with {len(payload['questions'])} questions")
    print(f"📄 Document URL: {payload['documents']}")
    print("=" * 60)
    
    # Test 1: Cache Miss (First Run)
    print("--- Testing First Run (Cache Miss) ---")
    print("⏳ Sending request... (this may take a while for document processing)")
    
    try:
        response_1, elapsed_1 = make_api_request(payload, timeout=180)
        print_response_details(response_1, elapsed_1, "Cache Miss")
        
    except Exception as e:
        print(f"❌ First run failed: {e}")
        return
    
    # Wait between requests to ensure first request is fully settled
    print("⏸️  Waiting 5 seconds for request to settle...")
    time.sleep(5)
    
    # Test 2: Cache Hit (Second Run)
    print("--- Testing Second Run (Cache Hit) ---")
    print("⏳ Sending identical request... (should be much faster with caching)")
    
    try:
        response_2, elapsed_2 = make_api_request(payload, timeout=180)
        print_response_details(response_2, elapsed_2, "Cache Hit")
        
    except Exception as e:
        print(f"❌ Second run failed: {e}")
        return
    
    # Performance comparison
    print("=" * 60)
    print("📈 Performance Analysis:")
    print(f"   Cache Miss Time:  {elapsed_1:.2f} seconds")
    print(f"   Cache Hit Time:   {elapsed_2:.2f} seconds")
    
    if elapsed_1 > elapsed_2:
        speedup = elapsed_1 / elapsed_2
        time_saved = elapsed_1 - elapsed_2
        print(f"   ⚡ Speedup:        {speedup:.2f}x faster")
        print(f"   💾 Time Saved:    {time_saved:.2f} seconds")
        print(f"   ✅ Caching is working effectively!")
    else:
        print(f"   ⚠️  Cache hit was not faster - this may indicate caching issues")
    
    print("=" * 60)
    print("🏁 Latency test completed successfully!")


def test_health_check():
    """
    Quick health check before running main tests.
    """
    print("🔍 Performing health check...")
    
    try:
        health_response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API server is healthy and ready")
            return True
        else:
            print(f"⚠️  Health check returned status {health_response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server - make sure it's running on http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


if __name__ == "__main__":
    # Run health check first
    if test_health_check():
        print()
        test_api_latency()
    else:
        print("\n💡 Make sure to start the FastAPI server first:")
        print("   python main.py")
        print("   or")
        print("   uvicorn main:app --reload")