# AstrID API Documentation

Welcome to the AstrID API documentation. This comprehensive guide provides everything you need to integrate with the AstrID astronomical identification system.

## 📚 Documentation Index

### Getting Started
- [Quick Start Guide](usage-guide.md#getting-started) - Get up and running in minutes
- [Authentication](usage-guide.md#authentication) - Learn how to authenticate with the API
- [API Versioning](usage-guide.md#api-versioning) - Understand our versioning strategy
- [Rate Limiting](usage-guide.md#rate-limiting) - Learn about request limits and best practices

### API Reference
- [OpenAPI Specification](openapi.yaml) - Complete API specification in OpenAPI 3.0 format
- [Interactive Documentation](http://localhost:8000/docs) - Try the API directly in your browser
- [ReDoc Documentation](http://localhost:8000/redoc) - Alternative documentation format

### Code Examples
- [Python Examples](usage-guide.md#python) - Complete Python client implementation
- [JavaScript Examples](usage-guide.md#javascriptnodejs) - Node.js client examples
- [cURL Examples](usage-guide.md#curl-examples) - Command-line examples

### Testing
- [API Testing Guide](testing-guide.md) - How to test the API
- [Test Scripts](../scripts/test-api.sh) - Automated testing scripts
- [Test Configuration](../tests/api/) - Test configuration and fixtures

## 🚀 Quick Start

### 1. Authentication

```bash
# Register a new user
curl -X POST "http://localhost:8000/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "password": "your-secure-password",
    "role": "user"
  }'

# Login to get access token
curl -X POST "http://localhost:8000/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "password": "your-secure-password"
  }'
```

### 2. Create an Observation

```bash
curl -X POST "http://localhost:8000/v1/observations" \
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
```

### 3. Run Detection

```bash
curl -X POST "http://localhost:8000/v1/detections/infer" \
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

## 📋 API Endpoints Overview

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/observations` | GET, POST | Manage astronomical observations |
| `/detections` | GET, POST | Run ML detection and manage results |
| `/workflows` | GET, POST | Orchestrate processing workflows |
| `/health` | GET | System health and status |

### Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register` | POST | Register new user |
| `/auth/login` | POST | User login |
| `/auth/refresh` | POST | Refresh access token |
| `/auth/logout` | POST | User logout |
| `/auth/profile` | GET, PUT | User profile management |

### Workflow Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/workflows/flows` | GET | List available workflows |
| `/workflows/flows/{type}/start` | POST | Start a workflow |
| `/workflows/flows/{id}/status` | GET | Get workflow status |
| `/workflows/flows/{id}/cancel` | POST | Cancel workflow |
| `/workflows/flows/{id}/logs` | GET | Get workflow logs |

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_BASE_URL=http://localhost:8000
API_VERSION=v1
ACCESS_TOKEN=your-access-token

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Rate Limits

| Category | Limit | Window |
|----------|-------|--------|
| Default | 1,000 requests | 1 hour |
| Observations | 100 requests | 1 hour |
| Detections | 200 requests | 1 hour |
| Workflows | 50 requests | 1 hour |
| Admin | 10,000 requests | 1 hour |

## 🧪 Testing

### Run API Tests

```bash
# Run all API tests
./scripts/test-api.sh

# Test specific environment
./scripts/test-api.sh -u https://api.astrid.chrislawrence.ca -t your-token

# Run with verbose output
./scripts/test-api.sh --verbose
```

### Test Categories

- **Unit Tests**: Individual endpoint testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Authentication and authorization testing
- **Rate Limiting Tests**: Rate limit enforcement testing

## 📊 Monitoring and Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/v1/health
```

### Metrics

The API provides comprehensive metrics including:
- Request/response times
- Error rates
- Rate limit usage
- Database connection status
- Service dependencies status

### Logging

All API requests and responses are logged with:
- Request ID for tracing
- User authentication status
- Response times
- Error details
- Rate limit information

## 🔒 Security

### Authentication
- JWT-based authentication
- Role-based access control (RBAC)
- Token refresh mechanism
- Secure password requirements

### Authorization
- Endpoint-level permissions
- Resource-level access control
- Admin user privileges
- API key management

### Data Protection
- HTTPS enforcement in production
- Input validation and sanitization
- SQL injection prevention
- XSS protection

## 🚀 Deployment

### Docker

```bash
# Build API image
docker build -t astrid-api .

# Run API container
docker run -p 8000:8000 astrid-api
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
```

### Production Considerations

- Use environment-specific configuration
- Enable HTTPS with valid certificates
- Configure proper CORS settings
- Set up monitoring and alerting
- Implement backup and recovery procedures

## 📞 Support

### Getting Help

- **Documentation**: [https://docs.astrid.chrislawrence.ca](https://docs.astrid.chrislawrence.ca)
- **API Reference**: [https://api.astrid.chrislawrence.ca/docs](https://api.astrid.chrislawrence.ca/docs)
- **Support Email**: astronomical.identification@gmail.com
- **GitHub Issues**: [https://github.com/astrid-project/issues](https://github.com/astrid-project/issues)

### Community

- **Discord**: [https://discord.gg/astrid](https://discord.gg/astrid)
- **GitHub Discussions**: [https://github.com/astrid-project/discussions](https://github.com/astrid-project/discussions)
- **Stack Overflow**: Tag questions with `astrid-api`

## 📝 Changelog

### Version 1.0.0 (Current)
- Initial API release
- Complete observation management
- ML detection pipeline
- Workflow orchestration
- Comprehensive documentation
- Rate limiting and versioning

### Upcoming Features
- Real-time streaming endpoints
- Advanced filtering and search
- Batch processing improvements
- Enhanced monitoring dashboard
- GraphQL API support

## 📄 License

This API is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

---

**Last Updated**: January 2025  
**API Version**: 1.0.0  
**Documentation Version**: 1.0.0
