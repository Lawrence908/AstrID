"""API versioning middleware and utilities."""

from typing import Optional, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


class APIVersioningMiddleware(BaseHTTPMiddleware):
    """Middleware for handling API versioning strategies."""
    
    def __init__(self, app, default_version: str = "v1"):
        super().__init__(app)
        self.default_version = default_version
        self.supported_versions = ["v1", "v2"]
    
    async def dispatch(self, request: Request, call_next):
        """Process request and extract version information."""
        
        # Extract version from multiple sources
        version = self._extract_version(request)
        
        # Add version to request state
        request.state.api_version = version
        
        # Validate version
        if version not in self.supported_versions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported API version: {version}. Supported versions: {', '.join(self.supported_versions)}"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add version headers to response
        response.headers["X-API-Version"] = version
        response.headers["X-API-Supported-Versions"] = ", ".join(self.supported_versions)
        
        return response
    
    def _extract_version(self, request: Request) -> str:
        """Extract API version from request using multiple strategies."""
        
        # Strategy 1: URL-based versioning (/api/v1/observations)
        path_parts = request.url.path.split('/')
        if len(path_parts) >= 3 and path_parts[1] == 'api' and path_parts[2].startswith('v'):
            return path_parts[2]
        
        # Strategy 2: Header-based versioning (Accept: application/vnd.astrid.v1+json)
        accept_header = request.headers.get('accept', '')
        if 'application/vnd.astrid.v' in accept_header:
            # Extract version from Accept header
            for part in accept_header.split(','):
                part = part.strip()
                if 'application/vnd.astrid.v' in part:
                    version_part = part.split('application/vnd.astrid.v')[1].split('+')[0]
                    if version_part.startswith('v'):
                        return version_part
                    else:
                        return f"v{version_part}"
        
        # Strategy 3: Query parameter versioning (?version=1)
        version_param = request.query_params.get('version')
        if version_param:
            if version_param.isdigit():
                return f"v{version_param}"
            elif version_param.startswith('v'):
                return version_param
        
        # Strategy 4: Custom header (X-API-Version)
        custom_header = request.headers.get('x-api-version')
        if custom_header:
            if custom_header.isdigit():
                return f"v{custom_header}"
            elif custom_header.startswith('v'):
                return custom_header
        
        # Default to configured default version
        return self.default_version


def get_api_version(request: Request) -> str:
    """Get the API version from request state."""
    return getattr(request.state, 'api_version', 'v1')


def validate_version_compatibility(current_version: str, required_version: str) -> bool:
    """Validate if current version is compatible with required version."""
    
    # Simple version comparison (can be enhanced for semantic versioning)
    current_major = current_version.lstrip('v').split('.')[0]
    required_major = required_version.lstrip('v').split('.')[0]
    
    return current_major == required_major


def get_versioned_path(path: str, version: str) -> str:
    """Convert path to versioned path."""
    
    # Remove existing version if present
    if '/api/v' in path:
        path = path.split('/api/v')[0] + path.split('/api/v')[1].split('/', 1)[1]
    
    # Add version
    if path.startswith('/api/'):
        return path.replace('/api/', f'/api/{version}/')
    elif path.startswith('/'):
        return f'/api/{version}{path}'
    else:
        return f'/api/{version}/{path}'


def create_version_deprecation_warning(version: str, deprecation_date: str, sunset_date: str) -> dict:
    """Create deprecation warning for API version."""
    
    return {
        "warning": "API version deprecation",
        "deprecated_version": version,
        "deprecation_date": deprecation_date,
        "sunset_date": sunset_date,
        "message": f"API version {version} is deprecated and will be removed on {sunset_date}. Please upgrade to a supported version."
    }


class VersionCompatibilityError(Exception):
    """Exception raised when API version compatibility check fails."""
    
    def __init__(self, current_version: str, required_version: str):
        self.current_version = current_version
        self.required_version = required_version
        super().__init__(
            f"API version {current_version} is not compatible with required version {required_version}"
        )
