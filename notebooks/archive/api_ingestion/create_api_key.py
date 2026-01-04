#!/usr/bin/env python3
"""Script to create API keys for automated workflows."""

import os
import sys

import requests

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

API_BASE = "http://localhost:9001"


def authenticate_user(email: str, password: str) -> str | None:
    """Authenticate user and get JWT token."""
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


def create_api_key(
    name: str,
    description: str = None,
    permission_set: str = "training_pipeline",
    expires_in_days: int = None,
    token: str = None,
) -> str | None:
    """Create a new API key."""
    if not token:
        print("❌ No authentication token provided")
        return None

    url = f"{API_BASE}/api-keys/"
    payload = {
        "name": name,
        "description": description,
        "permission_set": permission_set,
    }

    if expires_in_days:
        payload["expires_in_days"] = expires_in_days

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    print(f"Creating API key: {name}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        api_key_data = result.get("data", {})

        new_api_key = api_key_data.get("key")
        print("✅ API key created successfully!")
        print(f"   - Name: {api_key_data.get('name')}")
        print(f"   - Key: {new_api_key}")
        print(f"   - Permissions: {api_key_data.get('permissions')}")
        print(f"   - Expires: {api_key_data.get('expires_at', 'Never')}")

        return new_api_key

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to create API key: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error details: {error_detail}")
            except Exception:
                print(f"   Response: {e.response.text}")
        return None


def list_permission_sets() -> None:
    """List available permission sets."""
    url = f"{API_BASE}/api-keys/permission-sets"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        result = response.json()
        permission_sets = result.get("data", {})

        print("Available permission sets:")
        for name, permissions in permission_sets.items():
            print(f"  - {name}: {permissions}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to list permission sets: {str(e)}")


def main():
    """Main function to create API keys interactively."""
    print("=" * 60)
    print("ASTRID API KEY CREATOR")
    print("=" * 60)

    # Get user credentials
    email = input("Email: ").strip()
    password = input("Password: ").strip()

    # Authenticate
    token = authenticate_user(email, password)
    if not token:
        print("❌ Authentication failed. Cannot proceed.")
        return

    print("\n" + "=" * 40)
    print("AVAILABLE PERMISSION SETS")
    print("=" * 40)
    list_permission_sets()

    print("\n" + "=" * 40)
    print("CREATE API KEY")
    print("=" * 40)

    # Get API key details
    name = input("API Key Name: ").strip()
    description = input("Description (optional): ").strip() or None
    permission_set = (
        input("Permission Set (default: training_pipeline): ").strip()
        or "training_pipeline"
    )

    expires_input = input(
        "Expires in days (optional, press Enter for no expiration): "
    ).strip()
    expires_in_days = int(expires_input) if expires_input else None

    # Create the API key
    api_key = create_api_key(
        name=name,
        description=description,
        permission_set=permission_set,
        expires_in_days=expires_in_days,
        token=token,
    )

    if api_key:
        print("\n" + "=" * 40)
        print("API KEY CREATED SUCCESSFULLY")
        print("=" * 40)
        print(f"API Key: {api_key}")
        print("\nTo use this API key in your workflows:")
        print(f'  authenticate_with_api_key("{api_key}")')
        print("\nOr set as environment variable:")
        print(f'  export ASTRID_API_KEY="{api_key}"')
    else:
        print("❌ Failed to create API key")


if __name__ == "__main__":
    main()
