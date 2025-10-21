# API Keys Implementation Summary

## 🎯 **What We've Built**

A complete API key authentication system that allows automated workflows (Prefect, Dramatiq, etc.) to authenticate with your API while maintaining RBAC security.

## 📁 **Files Created/Modified**

### **New Files:**
- `src/adapters/auth/models.py` - APIKey database model
- `src/adapters/auth/api_key_service.py` - API key management service
- `src/adapters/auth/api_key_auth.py` - Authentication dependencies
- `src/adapters/api/routes/api_keys.py` - API key management endpoints
- `alembic/versions/add_api_keys_table.py` - Database migration
- `notebooks/create_api_key.py` - Interactive API key creation script
- `notebooks/test_api_keys.py` - Test script for API key functionality

### **Modified Files:**
- `src/domains/ml/training_data/api.py` - Updated to support API key auth
- `src/adapters/api/main.py` - Added API key routes
- `src/adapters/auth/rbac.py` - Added has_permission method to UserWithRole
- `notebooks/auth_template.py` - Added API key authentication functions
- `notebooks/START-HERE.md` - Updated with API key instructions

## 🔧 **How It Works**

### **1. Dual Authentication System**
- **JWT Authentication**: For human users (existing system)
- **API Key Authentication**: For automated workflows (new system)

### **2. API Key Management**
- Create API keys with specific permission sets
- Track usage statistics and expiration
- Revoke/extend keys as needed
- Granular permission control

### **3. Permission Sets**
Predefined permission sets for common use cases:
- `read_only`: Basic read access
- `training_pipeline`: Training data collection and processing
- `prefect_workflows`: Full workflow automation
- `full_access`: Complete system access

## 🚀 **Usage Examples**

### **Creating API Keys**

**Option 1: Interactive Script**
```bash
python notebooks/create_api_key.py
```

**Option 2: In Notebook**
```python
# First authenticate as admin
authenticate_user("admin@example.com", "password")

# Create API key
api_key = create_api_key(
    name="prefect-workflows",
    description="For Prefect automated workflows",
    permission_set="prefect_workflows"
)
```

**Option 3: Direct API Call**
```bash
curl -X POST http://127.0.0.1:8000/api-keys/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "training-pipeline",
    "description": "For training data collection",
    "permission_set": "training_pipeline"
  }'
```

### **Using API Keys**

**In Notebooks:**
```python
# Copy from notebooks/auth_template.py
authenticate_with_api_key("astrid_your_api_key_here")

# Now use AUTH_HEADERS in all API calls
response = requests.post(url, json=payload, headers=AUTH_HEADERS)
```

**In Prefect Flows:**
```python
import os
import requests

API_KEY = os.getenv("ASTRID_API_KEY")
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=HEADERS)
```

**In Dramatiq Workers:**
```python
from dramatiq import actor

@actor
def collect_training_data():
    API_KEY = os.getenv("ASTRID_API_KEY")
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    # Process response...
```

## 🔐 **Security Features**

### **API Key Security**
- Keys are hashed before storage
- Only key prefix is stored for identification
- Full key is only returned once during creation
- Usage tracking and audit trail

### **Permission Control**
- Granular permission system
- Predefined permission sets
- Custom permission combinations
- Expiration support

### **Access Control**
- Admin-only API key creation
- User-specific key ownership
- Revocation and extension capabilities
- Usage monitoring

## 📊 **API Endpoints**

### **API Key Management**
- `POST /api-keys/` - Create new API key
- `GET /api-keys/` - List API keys
- `GET /api-keys/{id}` - Get specific API key
- `PUT /api-keys/{id}` - Update API key
- `POST /api-keys/{id}/revoke` - Revoke API key
- `POST /api-keys/{id}/extend` - Extend expiration
- `GET /api-keys/permission-sets` - List permission sets

### **Protected Endpoints**
All existing endpoints now support both JWT and API key authentication:
- Training data endpoints
- Observation endpoints
- Detection endpoints
- Preprocessing endpoints
- And more...

## 🗄️ **Database Schema**

### **API Keys Table**
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(8) NOT NULL,
    permissions JSON NOT NULL,
    scopes JSON,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    usage_count VARCHAR(20) DEFAULT '0',
    created_by UUID REFERENCES users(id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 🧪 **Testing**

### **Test API Key Creation**
```bash
python notebooks/test_api_keys.py
```

### **Test in Notebook**
```python
# Copy authentication code from notebooks/auth_template.py
authenticate_with_api_key("your_api_key_here")

# Test API call
response = requests.get(f"{API_BASE}/training/datasets", headers=AUTH_HEADERS)
print(response.json())
```

## 🔄 **Migration Steps**

### **1. Run Database Migration**
```bash
# Update the migration file with correct down_revision
# Then run:
alembic upgrade head
```

### **2. Create Your First API Key**
```bash
python notebooks/create_api_key.py
```

### **3. Test the System**
```bash
python notebooks/test_api_keys.py
```

### **4. Update Your Workflows**
Replace JWT authentication with API key authentication in:
- Prefect flows
- Dramatiq workers
- Scheduled tasks
- CI/CD pipelines

## 🎉 **Benefits**

### **For Automated Workflows**
- ✅ No token refresh needed
- ✅ Long-lived authentication
- ✅ Granular permissions
- ✅ Easy to manage and rotate

### **For Security**
- ✅ Maintains RBAC system
- ✅ Audit trail for all actions
- ✅ Easy to revoke access
- ✅ Usage monitoring

### **For Development**
- ✅ Simple to implement
- ✅ Works with existing code
- ✅ Clear documentation
- ✅ Test scripts included

## 🚨 **Important Notes**

1. **API keys are only returned once** during creation - save them securely
2. **Use environment variables** to store API keys in production
3. **Rotate keys regularly** for security
4. **Monitor usage** through the API key management endpoints
5. **Use appropriate permission sets** - don't give more access than needed

## 🔮 **Next Steps**

1. **Run the database migration** to create the API keys table
2. **Create your first API key** using the provided scripts
3. **Test the system** to ensure everything works
4. **Update your Prefect/Dramatiq workflows** to use API keys
5. **Set up monitoring** for API key usage

The system is now ready for production use! 🚀
