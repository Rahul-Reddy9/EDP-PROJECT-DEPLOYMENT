"""
test_full_system_integration.py
--------------------------------
System smoke test verifying APIClient and core configuration.
"""

from frontend.api_client import APIClient


def test_api_client_initialization_and_url_resolution():
    """Verifies that the API client initializes with the proper default Render URL."""
    client = APIClient()
    assert "onrender.com" in client.base_url or "localhost" in client.base_url
    assert client.timeout == 10.0
