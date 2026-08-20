import sys
import argparse
import requests

def parse_args():
    parser = argparse.ArgumentParser(description="Render Free Tier Keep-Alive Pinger.")
    parser.add_argument("--url", type=str, required=True, help="Base URL of the deployed backend (e.g. https://xxxx.onrender.com).")
    return parser.parse_args()

def main():
    args = parse_args()
    base_url = args.url.rstrip('/')
    
    print(f"Pinging Render instance at {base_url} to maintain warm state...")
    
    # Ping health endpoint
    try:
        health_res = requests.get(f"{base_url}/api/health", timeout=10.0)
        print(f"Health check status: {health_res.status_code}")
        print(f"Health response: {health_res.text}")
    except Exception as e:
        print(f"Error pinging health endpoint: {e}")

    # Ping warmup endpoint
    try:
        warmup_res = requests.get(f"{base_url}/api/warmup", timeout=10.0)
        print(f"Warmup status: {warmup_res.status_code}")
        print(f"Warmup response: {warmup_res.text}")
    except Exception as e:
        print(f"Error pinging warmup endpoint: {e}")

if __name__ == "__main__":
    main()
