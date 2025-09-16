#!/usr/bin/env python3
"""
Script to wait for Prefect server to be ready before starting the worker.
This helps prevent connection errors during container startup.
"""

import sys
import time
from urllib.parse import urlparse

import requests


def wait_for_prefect_server(api_url: str, max_retries: int = 30, delay: float = 2.0):
    """
    Wait for Prefect server to be ready.

    Args:
        api_url: Prefect API URL (e.g., http://prefect:4200/api)
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    # Extract base URL from API URL
    parsed = urlparse(api_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    health_url = f"{base_url}/health"

    print(f"Waiting for Prefect server at {base_url}...")

    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"Prefect server is ready! (attempt {attempt + 1})")
                return True
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            print(
                f"Attempt {attempt + 1}/{max_retries}: Prefect server not ready - {e}"
            )

        if attempt < max_retries - 1:
            time.sleep(delay)

    print(f"Failed to connect to Prefect server after {max_retries} attempts")
    return False


if __name__ == "__main__":
    import os

    api_url = os.getenv("PREFECT_API_URL", "http://prefect:4200/api")
    max_retries = int(os.getenv("PREFECT_WAIT_MAX_RETRIES", "30"))
    delay = float(os.getenv("PREFECT_WAIT_DELAY", "2.0"))

    if not wait_for_prefect_server(api_url, max_retries, delay):
        sys.exit(1)

    print("Starting Prefect worker...")
    # The actual worker command will be executed by the shell
    sys.exit(0)
