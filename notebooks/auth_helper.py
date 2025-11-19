#!/usr/bin/env python3
"""
Authentication Helper for AstrID Notebooks

This script helps you authenticate and get a JWT token for use in notebooks.
"""

import requests

API_BASE = "http://localhost:9001"


def authenticate_user(email: str, password: str) -> str | None:
    """Authenticate user and return JWT token."""
    url = f"{API_BASE}/signin"
    payload = {"email": email, "password": password}

    print(f"Authenticating user: {email}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        auth_data = result.get("data", {})

        token = auth_data.get("access_token")
        user_info = auth_data.get("user", {})

        print(f"✅ Successfully authenticated as: {user_info.get('email', 'Unknown')}")
        print(f"   - User ID: {user_info.get('id', 'Unknown')}")
        print(f"   - Role: {user_info.get('role', 'Unknown')}")
        print(f"   - JWT Token: {token[:10]}...")

        return token

    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication failed: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error details: {error_detail}")
            except Exception:
                print(f"   Response: {e.response.text}")
        return None


def main():
    """Interactive authentication."""
    print("=" * 60)
    print("AstrID Authentication Helper")
    print("=" * 60)

    email = input("Email: ").strip()
    password = input("Password: ").strip()

    token = authenticate_user(email, password)

    if token:
        print("\n" + "=" * 60)
        print("AUTHENTICATION SUCCESSFUL!")
        print("=" * 60)
        print(f"\nJWT Token: {token}")
        print("\nUse this token in your notebooks:")
        print(f'AUTH_TOKEN = "{token}"')
        print("AUTH_HEADERS = {")
        print(f'    "Authorization": f"Bearer {token}",')
        print('    "Content-Type": "application/json"')
        print("}")
        print("\nOr call authenticate_user(email, password) in your notebook.")
    else:
        print("\n❌ Authentication failed. Please check your credentials.")


if __name__ == "__main__":
    main()
