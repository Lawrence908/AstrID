#!/usr/bin/env python3
"""Test script for API key functionality."""

import os
import sys

import requests

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

API_BASE = "http://127.0.0.1:8000"


def test_api_key_creation():
    """Test creating an API key."""
    print("=" * 60)
    print("TESTING API KEY CREATION")
    print("=" * 60)

    # First, authenticate as a user
    email = input("Email: ").strip()
    password = input("Password: ").strip()

    # Authenticate
    url = f"{API_BASE}/signin"
    payload = {"email": email, "password": password}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        auth_data = result.get("data", {})
        token = auth_data.get("access_token")

        if not token:
            print("❌ No token received")
            return None

        print("✅ Authenticated successfully")

        # Create API key
        url = f"{API_BASE}/api-keys/"
        payload = {
            "name": "test-key",
            "description": "Test API key",
            "permission_set": "training_pipeline",
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        api_key_data = result.get("data", {})

        api_key = api_key_data.get("key")
        print(f"✅ API key created: {api_key}")

        return api_key

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            print(f"   Response: {e.response.text}")
        return None


def test_api_key_usage(api_key: str):
    """Test using an API key to access protected endpoints."""
    print("\n" + "=" * 60)
    print("TESTING API KEY USAGE")
    print("=" * 60)

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    # Test training datasets endpoint
    url = f"{API_BASE}/training/datasets"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        print("✅ Successfully accessed training datasets endpoint")
        print(f"   Response: {result}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error accessing training datasets: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            print(f"   Response: {e.response.text}")


def main():
    """Main test function."""
    print("API Key System Test")
    print("This will test creating and using API keys")

    # Test API key creation
    api_key = test_api_key_creation()

    if api_key:
        # Test API key usage
        test_api_key_usage(api_key)

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f"API Key: {api_key}")
        print("\nTo use this API key in your workflows:")
        print(f'  authenticate_with_api_key("{api_key}")')
    else:
        print("❌ Test failed - could not create API key")


if __name__ == "__main__":
    main()
