# AstrID API Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [API Versioning](#api-versioning)
4. [Rate Limiting](#rate-limiting)
5. [Error Handling](#error-handling)
6. [Common Workflows](#common-workflows)
7. [Code Examples](#code-examples)
8. [Best Practices](#best-practices)

## Getting Started

The AstrID API provides comprehensive endpoints for astronomical observation management, anomaly detection, and workflow orchestration. This guide will help you get started with the API.

### Base URLs

- **Production**: `https://api.astrid.chrislawrence.ca/v1`
- **Staging**: `https://staging-api.astrid.chrislawrence.ca/v1`
- **Development**: `http://localhost:8000/v1`

### Quick Start

1. **Register for an account**:
   ```bash
   curl -X POST "https://api.astrid.chrislawrence.ca/v1/auth/register" \
     -H "Content-Type: application/json" \
     -d '{
       "email": "your-email@example.com",
       "password": "your-secure-password",
       "role": "user"
     }'
   ```

2. **Login to get access token**:
   ```bash
   curl -X POST "https://api.astrid.chrislawrence.ca/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "email": "your-email@example.com",
       "password": "your-secure-password"
     }'
   ```

3. **Use the access token for authenticated requests**:
   ```bash
   curl -X GET "https://api.astrid.chrislawrence.ca/v1/observations" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
   ```

## Authentication

The AstrID API uses JWT (JSON Web Token) based authentication with role-based access control (RBAC).

### Authentication Flow

1. **Register** a new user account
2. **Login** to receive access and refresh tokens
3. **Include the access token** in the Authorization header for protected endpoints
4. **Refresh the token** when it expires using the refresh token

### Token Management

```python
import requests
import time

class AstrIDClient:
    def __init__(self, base_url, email, password):
        self.base_url = base_url
        self.email = email
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def authenticate(self):
        """Authenticate and get access token."""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"email": self.email, "password": self.password}
        )
        
        if response.status_code == 200:
            data = response.json()["data"]
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            # Token typically expires in 30 minutes
            self.token_expires_at = time.time() + 1800
            return True
        return False
    
    def get_headers(self):
        """Get headers with authentication."""
        if not self.access_token or time.time() >= self.token_expires_at:
            self.refresh_access_token()
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def refresh_access_token(self):
        """Refresh the access token using refresh token."""
        response = requests.post(
            f"{self.base_url}/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )
        
        if response.status_code == 200:
            data = response.json()["data"]
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.token_expires_at = time.time() + 1800
```

## API Versioning

The AstrID API supports multiple versioning strategies:

### URL-based Versioning

```
GET /api/v1/observations
POST /api/v2/observations
```

### Header-based Versioning

```bash
curl -H "Accept: application/vnd.astrid.v1+json" \
     "https://api.astrid.chrislawrence.ca/observations"
```

### Query Parameter Versioning

```bash
curl "https://api.astrid.chrislawrence.ca/observations?version=1"
```

### Version Compatibility

- **v1**: Current stable version
- **v2**: Next major version (in development)
- **Backward compatibility**: v1 endpoints will remain supported for 12 months after v2 release

## Rate Limiting

API requests are rate-limited to ensure fair usage and system stability.

### Rate Limits by Endpoint

| Endpoint Category | Limit | Window |
|------------------|-------|--------|
| Default | 1,000 requests | 1 hour |
| Observations | 100 requests | 1 hour |
| Detections | 200 requests | 1 hour |
| Workflows | 50 requests | 1 hour |
| Admin | 10,000 requests | 1 hour |

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_retry(client, url, max_retries=3):
    """Make request with automatic retry on rate limit."""
    for attempt in range(max_retries):
        response = client.get(url)
        
        if response.status_code == 429:
            # Rate limited - wait and retry
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(1, reset_time - int(time.time()))
            time.sleep(wait_time)
            continue
        
        return response
    
    raise Exception("Max retries exceeded")
```

## Error Handling

All API responses follow a consistent error format:

### Success Response

```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "survey": "ZTF",
    "observation_id": "ZTF_20230101_000000"
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "message": "Resource not found",
    "error_code": "RESOURCE_NOT_FOUND",
    "details": "Observation with ID 123 not found",
    "type": "ResourceNotFoundError"
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_ERROR` | 401 | Invalid or missing authentication |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource doesn't exist |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `RESOURCE_CONFLICT` | 409 | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Server error |

## Common Workflows

### 1. Observation Processing Workflow

```python
# 1. Create observation
observation = client.post("/observations", json={
    "survey": "ZTF",
    "observation_id": "ZTF_20230101_000000",
    "ra": 180.5,
    "dec": 45.2,
    "observation_time": "2025-01-01T00:00:00Z",
    "filter_band": "r",
    "exposure_time": 30.0
})

# 2. Start preprocessing workflow
workflow = client.post("/workflows/flows/observation_processing/start", json={
    "observation_id": observation["data"]["id"],
    "preprocessing_enabled": True,
    "differencing_enabled": True,
    "detection_enabled": True
})

# 3. Monitor workflow status
status = client.get(f"/workflows/flows/{workflow['data']['flow_id']}/status")

# 4. Get detection results
detections = client.get("/detections", params={
    "observation_id": observation["data"]["id"]
})
```

### 2. Batch Detection Workflow

```python
# 1. Prepare batch data
batch_data = {
    "observations": [
        {
            "observation_id": "obs_1",
            "ra": 180.5,
            "dec": 45.2,
            "confidence": 0.95,
            "magnitude": 18.5,
            "model_version": "v1.0.0"
        },
        {
            "observation_id": "obs_2",
            "ra": 181.0,
            "dec": 46.0,
            "confidence": 0.85,
            "magnitude": 19.0,
            "model_version": "v1.0.0"
        }
    ]
}

# 2. Run batch detection
detections = client.post("/detections/batch", json=batch_data)

# 3. Validate detections
for detection in detections["data"]:
    client.put(f"/detections/{detection['id']}/validate", json={
        "status": "validated",
        "validator_notes": "Confirmed anomaly"
    })
```

## Code Examples

### Python

```python
import requests
import json
from typing import Dict, List, Optional

class AstrIDAPI:
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def create_observation(self, observation_data: Dict) -> Dict:
        """Create a new observation."""
        response = requests.post(
            f"{self.base_url}/observations",
            json=observation_data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_observations(self, 
                        page: int = 1, 
                        size: int = 20,
                        survey: Optional[str] = None) -> Dict:
        """Get observations with pagination and filtering."""
        params = {"page": page, "size": size}
        if survey:
            params["survey"] = survey
        
        response = requests.get(
            f"{self.base_url}/observations",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def run_detection(self, detection_data: Dict) -> Dict:
        """Run ML detection on observation."""
        response = requests.post(
            f"{self.base_url}/detections/infer",
            json=detection_data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
api = AstrIDAPI("https://api.astrid.chrislawrence.ca/v1", "your-access-token")

# Create observation
observation = api.create_observation({
    "survey": "ZTF",
    "observation_id": "ZTF_20230101_000000",
    "ra": 180.5,
    "dec": 45.2,
    "observation_time": "2025-01-01T00:00:00Z",
    "filter_band": "r",
    "exposure_time": 30.0
})

# Get observations
observations = api.get_observations(page=1, size=10, survey="ZTF")

# Run detection
detection = api.run_detection({
    "observation_id": observation["data"]["id"],
    "ra": 180.5,
    "dec": 45.2,
    "confidence": 0.95,
    "magnitude": 18.5,
    "model_version": "v1.0.0"
})
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class AstrIDAPI {
    constructor(baseUrl, accessToken) {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            }
        });
    }
    
    async createObservation(observationData) {
        const response = await this.client.post('/observations', observationData);
        return response.data;
    }
    
    async getObservations(page = 1, size = 20, survey = null) {
        const params = { page, size };
        if (survey) params.survey = survey;
        
        const response = await this.client.get('/observations', { params });
        return response.data;
    }
    
    async runDetection(detectionData) {
        const response = await this.client.post('/detections/infer', detectionData);
        return response.data;
    }
}

// Usage
const api = new AstrIDAPI('https://api.astrid.chrislawrence.ca/v1', 'your-access-token');

async function processObservation() {
    try {
        // Create observation
        const observation = await api.createObservation({
            survey: 'ZTF',
            observation_id: 'ZTF_20230101_000000',
            ra: 180.5,
            dec: 45.2,
            observation_time: '2025-01-01T00:00:00Z',
            filter_band: 'r',
            exposure_time: 30.0
        });
        
        // Get observations
        const observations = await api.getObservations(1, 10, 'ZTF');
        
        // Run detection
        const detection = await api.runDetection({
            observation_id: observation.data.id,
            ra: 180.5,
            dec: 45.2,
            confidence: 0.95,
            magnitude: 18.5,
            model_version: 'v1.0.0'
        });
        
        console.log('Detection completed:', detection);
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}
```

### cURL Examples

```bash
# Create observation
curl -X POST "https://api.astrid.chrislawrence.ca/v1/observations" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "survey": "ZTF",
    "observation_id": "ZTF_20230101_000000",
    "ra": 180.5,
    "dec": 45.2,
    "observation_time": "2025-01-01T00:00:00Z",
    "filter_band": "r",
    "exposure_time": 30.0
  }'

# Get observations with pagination
curl -X GET "https://api.astrid.chrislawrence.ca/v1/observations?page=1&size=20&survey=ZTF" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Run detection
curl -X POST "https://api.astrid.chrislawrence.ca/v1/detections/infer" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "observation_id": "123e4567-e89b-12d3-a456-426614174000",
    "ra": 180.5,
    "dec": 45.2,
    "confidence": 0.95,
    "magnitude": 18.5,
    "model_version": "v1.0.0"
  }'
```

## Best Practices

### 1. Authentication
- Store access tokens securely
- Implement automatic token refresh
- Handle authentication errors gracefully

### 2. Rate Limiting
- Monitor rate limit headers
- Implement exponential backoff for retries
- Use batch endpoints when possible

### 3. Error Handling
- Always check response status codes
- Implement proper error logging
- Provide meaningful error messages to users

### 4. Performance
- Use pagination for large datasets
- Implement caching where appropriate
- Use appropriate HTTP methods (GET for reads, POST for creates)

### 5. Security
- Never expose access tokens in client-side code
- Use HTTPS in production
- Validate all input data

### 6. Monitoring
- Log API requests and responses
- Monitor rate limit usage
- Track error rates and response times

## Support

For additional help:

- **Documentation**: [https://docs.astrid.chrislawrence.ca](https://docs.astrid.chrislawrence.ca)
- **API Reference**: [https://api.astrid.chrislawrence.ca/docs](https://api.astrid.chrislawrence.ca/docs)
- **Support Email**: astronomical.identification@gmail.com
- **GitHub Issues**: [https://github.com/Lawrence908/AstrID/issues](https://github.com/Lawrence908/AstrID/issues)
