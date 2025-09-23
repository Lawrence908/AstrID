#!/bin/bash

# AstrID API Testing Script
# Comprehensive API testing with multiple test scenarios

set -e

# Configuration
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
API_VERSION="${API_VERSION:-v1}"
ACCESS_TOKEN="${ACCESS_TOKEN:-}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test result tracking
test_result() {
    local test_name="$1"
    local status="$2"
    local message="$3"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if [ "$status" = "PASS" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        log_success "$test_name: $message"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        log_error "$test_name: $message"
    fi
}

# HTTP request helper
make_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local headers="$4"

    local url="${API_BASE_URL}/${API_VERSION}${endpoint}"
    local curl_cmd="curl -s -w '%{http_code}'"

    if [ -n "$headers" ]; then
        curl_cmd="$curl_cmd -H '$headers'"
    fi

    if [ "$method" = "POST" ] || [ "$method" = "PUT" ]; then
        curl_cmd="$curl_cmd -X $method -H 'Content-Type: application/json' -d '$data'"
    else
        curl_cmd="$curl_cmd -X $method"
    fi

    curl_cmd="$curl_cmd '$url'"

    if [ "$VERBOSE" = "true" ]; then
        log_info "Making request: $curl_cmd"
    fi

    eval "$curl_cmd"
}

# Test authentication
test_authentication() {
    log_info "Testing authentication endpoints..."

    # Test user registration
    local register_data='{"email":"test@example.com","password":"testpassword123","role":"user"}'
    local response=$(make_request "POST" "/auth/register" "$register_data")
    local status_code="${response: -3}"
    local body="${response%???}"

    if [ "$status_code" = "201" ]; then
        test_result "User Registration" "PASS" "User registered successfully"
        ACCESS_TOKEN=$(echo "$body" | jq -r '.data.access_token // empty')
    else
        test_result "User Registration" "FAIL" "Status: $status_code, Response: $body"
    fi

    # Test user login
    local login_data='{"email":"test@example.com","password":"testpassword123"}'
    response=$(make_request "POST" "/auth/login" "$login_data")
    status_code="${response: -3}"
    body="${response%???}"

    if [ "$status_code" = "200" ]; then
        test_result "User Login" "PASS" "User logged in successfully"
        if [ -z "$ACCESS_TOKEN" ]; then
            ACCESS_TOKEN=$(echo "$body" | jq -r '.data.access_token // empty')
        fi
    else
        test_result "User Login" "FAIL" "Status: $status_code, Response: $body"
    fi

    # Test protected endpoint access
    if [ -n "$ACCESS_TOKEN" ]; then
        local auth_header="Authorization: Bearer $ACCESS_TOKEN"
        response=$(make_request "GET" "/auth/profile" "" "$auth_header")
        status_code="${response: -3}"
        body="${response%???}"

        if [ "$status_code" = "200" ]; then
            test_result "Protected Endpoint Access" "PASS" "Access token valid"
        else
            test_result "Protected Endpoint Access" "FAIL" "Status: $status_code, Response: $body"
        fi
    else
        test_result "Protected Endpoint Access" "FAIL" "No access token available"
    fi
}

# Test observations API
test_observations() {
    log_info "Testing observations endpoints..."

    if [ -z "$ACCESS_TOKEN" ]; then
        test_result "Observations API" "FAIL" "No access token available"
        return
    fi

    local auth_header="Authorization: Bearer $ACCESS_TOKEN"

    # Test create observation
    local observation_data='{"survey":"ZTF","observation_id":"ZTF_20230101_000000","ra":180.5,"dec":45.2,"observation_time":"2025-01-01T00:00:00Z","filter_band":"r","exposure_time":30.0}'
    local response=$(make_request "POST" "/observations" "$observation_data" "$auth_header")
    local status_code="${response: -3}"
    local body="${response%???}"

    if [ "$status_code" = "201" ]; then
        test_result "Create Observation" "PASS" "Observation created successfully"
        local observation_id=$(echo "$body" | jq -r '.data.id // empty')

        if [ -n "$observation_id" ]; then
            # Test get observation
            response=$(make_request "GET" "/observations/$observation_id" "" "$auth_header")
            status_code="${response: -3}"
            body="${response%???}"

            if [ "$status_code" = "200" ]; then
                test_result "Get Observation" "PASS" "Observation retrieved successfully"
            else
                test_result "Get Observation" "FAIL" "Status: $status_code, Response: $body"
            fi

            # Test update observation status
            local status_data='{"status":"preprocessing"}'
            response=$(make_request "PUT" "/observations/$observation_id/status" "$status_data" "$auth_header")
            status_code="${response: -3}"
            body="${response%???}"

            if [ "$status_code" = "200" ]; then
                test_result "Update Observation Status" "PASS" "Status updated successfully"
            else
                test_result "Update Observation Status" "FAIL" "Status: $status_code, Response: $body"
            fi
        fi
    else
        test_result "Create Observation" "FAIL" "Status: $status_code, Response: $body"
    fi

    # Test list observations
    response=$(make_request "GET" "/observations?page=1&size=10" "" "$auth_header")
    status_code="${response: -3}"
    body="${response%???}"

    if [ "$status_code" = "200" ]; then
        test_result "List Observations" "PASS" "Observations listed successfully"
    else
        test_result "List Observations" "FAIL" "Status: $status_code, Response: $body"
    fi
}

# Test detections API
test_detections() {
    log_info "Testing detections endpoints..."

    if [ -z "$ACCESS_TOKEN" ]; then
        test_result "Detections API" "FAIL" "No access token available"
        return
    fi

    local auth_header="Authorization: Bearer $ACCESS_TOKEN"

    # Test run detection inference
    local detection_data='{"observation_id":"123e4567-e89b-12d3-a456-426614174000","ra":180.5,"dec":45.2,"confidence":0.95,"magnitude":18.5,"model_version":"v1.0.0"}'
    local response=$(make_request "POST" "/detections/infer" "$detection_data" "$auth_header")
    local status_code="${response: -3}"
    local body="${response%???}"

    if [ "$status_code" = "200" ]; then
        test_result "Run Detection Inference" "PASS" "Detection inference completed"
        local detection_id=$(echo "$body" | jq -r '.data.detection_id // empty')

        if [ -n "$detection_id" ]; then
            # Test get detection
            response=$(make_request "GET" "/detections/$detection_id" "" "$auth_header")
            status_code="${response: -3}"
            body="${response%???}"

            if [ "$status_code" = "200" ]; then
                test_result "Get Detection" "PASS" "Detection retrieved successfully"
            else
                test_result "Get Detection" "FAIL" "Status: $status_code, Response: $body"
            fi
        fi
    else
        test_result "Run Detection Inference" "FAIL" "Status: $status_code, Response: $body"
    fi

    # Test list detections
    response=$(make_request "GET" "/detections?page=1&size=10" "" "$auth_header")
    status_code="${response: -3}"
    body="${response%???}"

    if [ "$status_code" = "200" ]; then
        test_result "List Detections" "PASS" "Detections listed successfully"
    else
        test_result "List Detections" "FAIL" "Status: $status_code, Response: $body"
    fi
}

# Test workflows API
test_workflows() {
    log_info "Testing workflows endpoints..."

    if [ -z "$ACCESS_TOKEN" ]; then
        test_result "Workflows API" "FAIL" "No access token available"
        return
    fi

    local auth_header="Authorization: Bearer $ACCESS_TOKEN"

    # Test list flows
    local response=$(make_request "GET" "/workflows/flows" "" "$auth_header")
    local status_code="${response: -3}"
    local body="${response%???}"

    if [ "$status_code" = "200" ]; then
        test_result "List Flows" "PASS" "Flows listed successfully"
    else
        test_result "List Flows" "FAIL" "Status: $status_code, Response: $body"
    fi

    # Test start flow
    local flow_data='{"observation_id":"123e4567-e89b-12d3-a456-426614174000","preprocessing_enabled":true,"differencing_enabled":true,"detection_enabled":true}'
    response=$(make_request "POST" "/workflows/flows/observation_processing/start" "$flow_data" "$auth_header")
    status_code="${response: -3}"
    body="${response%???}"

    if [ "$status_code" = "200" ]; then
        test_result "Start Flow" "PASS" "Flow started successfully"
        local flow_id=$(echo "$body" | jq -r '.data.flow_id // empty')

        if [ -n "$flow_id" ]; then
            # Test get flow status
            response=$(make_request "GET" "/workflows/flows/$flow_id/status" "" "$auth_header")
            status_code="${response: -3}"
            body="${response%???}"

            if [ "$status_code" = "200" ]; then
                test_result "Get Flow Status" "PASS" "Flow status retrieved successfully"
            else
                test_result "Get Flow Status" "FAIL" "Status: $status_code, Response: $body"
            fi
        fi
    else
        test_result "Start Flow" "FAIL" "Status: $status_code, Response: $body"
    fi
}

# Test health endpoints
test_health() {
    log_info "Testing health endpoints..."

    # Test basic health check
    local response=$(make_request "GET" "/health" "")
    local status_code="${response: -3}"
    local body="${response%???}"

    if [ "$status_code" = "200" ] || [ "$status_code" = "503" ]; then
        test_result "Health Check" "PASS" "Health endpoint accessible"
    else
        test_result "Health Check" "FAIL" "Status: $status_code, Response: $body"
    fi

    # Test API info
    response=$(make_request "GET" "/" "")
    status_code="${response: -3}"
    body="${response%???}"

    if [ "$status_code" = "200" ]; then
        test_result "API Info" "PASS" "API info endpoint accessible"
    else
        test_result "API Info" "FAIL" "Status: $status_code, Response: $body"
    fi
}

# Test rate limiting
test_rate_limiting() {
    log_info "Testing rate limiting..."

    if [ -z "$ACCESS_TOKEN" ]; then
        test_result "Rate Limiting" "FAIL" "No access token available"
        return
    fi

    local auth_header="Authorization: Bearer $ACCESS_TOKEN"
    local rate_limited=false

    # Make many requests quickly to trigger rate limiting
    for i in $(seq 1 150); do
        local response=$(make_request "GET" "/observations" "" "$auth_header")
        local status_code="${response: -3}"

        if [ "$status_code" = "429" ]; then
            rate_limited=true
            break
        fi

        # Small delay to avoid overwhelming the server
        sleep 0.01
    done

    if [ "$rate_limited" = "true" ]; then
        test_result "Rate Limiting" "PASS" "Rate limiting working correctly"
    else
        test_result "Rate Limiting" "WARN" "Rate limiting not triggered (may be normal)"
    fi
}

# Test error handling
test_error_handling() {
    log_info "Testing error handling..."

    # Test invalid endpoint
    local response=$(make_request "GET" "/invalid-endpoint" "")
    local status_code="${response: -3}"

    if [ "$status_code" = "404" ]; then
        test_result "404 Error Handling" "PASS" "Invalid endpoint returns 404"
    else
        test_result "404 Error Handling" "FAIL" "Status: $status_code"
    fi

    # Test unauthorized access
    response=$(make_request "GET" "/observations" "")
    status_code="${response: -3}"

    if [ "$status_code" = "401" ]; then
        test_result "401 Error Handling" "PASS" "Unauthorized access returns 401"
    else
        test_result "401 Error Handling" "FAIL" "Status: $status_code"
    fi

    # Test validation error
    if [ -n "$ACCESS_TOKEN" ]; then
        local auth_header="Authorization: Bearer $ACCESS_TOKEN"
        local invalid_data='{"invalid_field":"invalid_value"}'
        response=$(make_request "POST" "/observations" "$invalid_data" "$auth_header")
        status_code="${response: -3}"

        if [ "$status_code" = "400" ] || [ "$status_code" = "422" ]; then
            test_result "Validation Error Handling" "PASS" "Invalid data returns validation error"
        else
            test_result "Validation Error Handling" "FAIL" "Status: $status_code"
        fi
    fi
}

# Main test execution
main() {
    log_info "Starting AstrID API tests..."
    log_info "API Base URL: $API_BASE_URL"
    log_info "API Version: $API_VERSION"

    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Please install jq to run API tests."
        exit 1
    fi

    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed. Please install curl to run API tests."
        exit 1
    fi

    # Run tests
    test_health
    test_authentication
    test_observations
    test_detections
    test_workflows
    test_rate_limiting
    test_error_handling

    # Print summary
    echo
    log_info "Test Summary:"
    log_info "Total Tests: $TOTAL_TESTS"
    log_success "Passed: $PASSED_TESTS"
    if [ $FAILED_TESTS -gt 0 ]; then
        log_error "Failed: $FAILED_TESTS"
    else
        log_success "Failed: $FAILED_TESTS"
    fi

    # Calculate success rate
    if [ $TOTAL_TESTS -gt 0 ]; then
        local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
        log_info "Success Rate: $success_rate%"

        if [ $success_rate -ge 80 ]; then
            log_success "API tests completed successfully!"
            exit 0
        else
            log_error "API tests failed. Success rate below 80%."
            exit 1
        fi
    else
        log_error "No tests were executed."
        exit 1
    fi
}

# Help function
show_help() {
    echo "AstrID API Testing Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -u, --url URL          API base URL (default: http://127.0.0.1:8000)"
    echo "  -v, --version VERSION  API version (default: v1)"
    echo "  -t, --token TOKEN      Access token for authenticated requests"
    echo "  --verbose              Enable verbose output"
    echo "  -h, --help             Show this help message"
    echo
    echo "Environment Variables:"
    echo "  API_BASE_URL           API base URL"
    echo "  API_VERSION            API version"
    echo "  ACCESS_TOKEN           Access token"
    echo "  VERBOSE                Enable verbose output"
    echo
    echo "Examples:"
    echo "  $0                                    # Test local API"
    echo "  $0 -u https://api.astrid.chrislawrence.ca         # Test production API"
    echo "  $0 -t your-access-token              # Test with access token"
    echo "  $0 --verbose                         # Enable verbose output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            API_BASE_URL="$2"
            shift 2
            ;;
        -v|--version)
            API_VERSION="$2"
            shift 2
            ;;
        -t|--token)
            ACCESS_TOKEN="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main
