# =============================================================================
# AUTHENTICATION TEMPLATE - COPY THIS TO ANY NOTEBOOK THAT NEEDS API AUTH
# =============================================================================


import requests

# Configuration
API_BASE = "http://127.0.0.1:8000"
AUTH_TOKEN: str | None = None
API_KEY: str | None = None
AUTH_HEADERS: dict = {}


def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user and get JWT token."""
    global AUTH_TOKEN, AUTH_HEADERS

    url = f"{API_BASE}/signin"
    payload = {"email": email, "password": password}

    print(f"Authenticating user: {email}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        auth_data = result.get("data", {})

        AUTH_TOKEN = auth_data.get("access_token")
        AUTH_HEADERS = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
        }

        user_info = auth_data.get("user", {})
        print(f"✅ Successfully authenticated as: {user_info.get('email', 'Unknown')}")
        print(f"   - User ID: {user_info.get('id', 'Unknown')}")
        print(f"   - Role: {user_info.get('role', 'Unknown')}")
        print(f"   - JWT Token: {AUTH_TOKEN[:50]}...")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication failed: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error details: {error_detail}")
            except Exception:
                print(f"   Response: {e.response.text}")
        return False


def authenticate_with_api_key(api_key: str) -> bool:
    """Authenticate using an API key."""
    global API_KEY, AUTH_HEADERS

    API_KEY = api_key
    AUTH_HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    print("✅ Using API key authentication")
    print(f"   - API Key: {API_KEY[:20]}...")

    return True


def create_api_key(
    name: str, description: str = None, permission_set: str = "training_pipeline"
) -> str:
    """Create a new API key (requires admin authentication)."""
    if not AUTH_TOKEN:
        print("❌ Must be authenticated with JWT to create API keys")
        return None

    url = f"{API_BASE}/api-keys/"
    payload = {
        "name": name,
        "description": description,
        "permission_set": permission_set,
    }

    print(f"Creating API key: {name}")

    try:
        response = requests.post(url, json=payload, headers=AUTH_HEADERS, timeout=30)
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


# Usage Examples:
#
# Option 1: JWT Authentication (for human users)
# authenticate_user("your-email@example.com", "your-password")
#
# Option 2: API Key Authentication (for automated workflows)
# authenticate_with_api_key("astrid_your_api_key_here")
#
# Option 3: Create API Key (requires admin JWT auth first)
# authenticate_user("admin@example.com", "password")
# api_key = create_api_key("prefect-workflows", "For Prefect automated workflows")
#
# Then use AUTH_HEADERS in all API calls:
# response = requests.post(url, json=payload, headers=AUTH_HEADERS)

print("🔐 Authentication template loaded - ready to use!")
