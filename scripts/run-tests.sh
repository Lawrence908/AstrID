#!/bin/bash

# AstrID Test Execution Script
# Runs tests with different configurations and generates coverage reports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=true
PARALLEL=false
VERBOSE=false
MARKERS=""
OUTPUT_DIR="htmlcov"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE        Test type: unit, integration, e2e, performance, all (default: all)"
            echo "  --no-coverage          Disable coverage reporting"
            echo "  -p, --parallel         Run tests in parallel"
            echo "  -v, --verbose          Verbose output"
            echo "  -m, --markers MARKERS  Run tests with specific markers"
            echo "  -o, --output DIR       Coverage output directory (default: htmlcov)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all tests with coverage"
            echo "  $0 -t unit --parallel        # Run unit tests in parallel"
            echo "  $0 -m \"not slow\"            # Run tests excluding slow ones"
            echo "  $0 -t performance -v         # Run performance tests with verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the project root
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [[ -f ".venv/bin/activate" ]]; then
    print_status "Activating virtual environment..."
    source .venv/bin/activate
fi

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add test path based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration"
        ;;
    e2e)
        PYTEST_CMD="$PYTEST_CMD tests/e2e"
        ;;
    performance)
        PYTEST_CMD="$PYTEST_CMD -m performance"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
    *)
        print_error "Invalid test type: $TEST_TYPE"
        exit 1
        ;;
esac

# Add coverage if enabled
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html:$OUTPUT_DIR --cov-report=term-missing --cov-report=xml"
fi

# Add parallel execution if enabled
if [[ "$PARALLEL" == true ]]; then
    # Detect number of CPU cores
    NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    PYTEST_CMD="$PYTEST_CMD -n $NUM_CORES"
    print_status "Running tests in parallel with $NUM_CORES workers"
fi

# Add verbose output if enabled
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add custom markers if specified
if [[ -n "$MARKERS" ]]; then
    PYTEST_CMD="$PYTEST_CMD -m \"$MARKERS\""
fi

# Add other useful options
PYTEST_CMD="$PYTEST_CMD --tb=short --strict-markers"

print_status "Running tests with command: $PYTEST_CMD"
print_status "Test type: $TEST_TYPE"
print_status "Coverage: $COVERAGE"
print_status "Parallel: $PARALLEL"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the tests
print_status "Starting test execution..."
start_time=$(date +%s)

if eval "$PYTEST_CMD" 2>&1 | tee logs/test-output.log; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    print_success "Tests completed successfully in ${duration}s"

    # Display coverage summary if enabled
    if [[ "$COVERAGE" == true ]]; then
        print_status "Coverage report generated in: $OUTPUT_DIR/index.html"

        # Extract coverage percentage from output
        if grep -q "TOTAL" logs/test-output.log; then
            coverage_pct=$(grep "TOTAL" logs/test-output.log | awk '{print $NF}')
            print_status "Total coverage: $coverage_pct"

            # Check if coverage meets threshold
            coverage_num=$(echo "$coverage_pct" | sed 's/%//')
            if (( $(echo "$coverage_num >= 90" | bc -l) )); then
                print_success "Coverage threshold met (≥90%)"
            else
                print_warning "Coverage below threshold (<90%): $coverage_pct"
            fi
        fi
    fi

    # Display test summary
    if grep -q "passed" logs/test-output.log; then
        test_summary=$(grep -E "=+ .* passed.*in.*=+" logs/test-output.log | tail -1)
        print_success "Test summary: $test_summary"
    fi

else
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    print_error "Tests failed after ${duration}s"
    print_error "Check logs/test-output.log for details"
    exit 1
fi

# Open coverage report if available and running on a system with a browser
if [[ "$COVERAGE" == true ]] && [[ -f "$OUTPUT_DIR/index.html" ]]; then
    if command -v xdg-open &> /dev/null; then
        print_status "Opening coverage report in browser..."
        xdg-open "$OUTPUT_DIR/index.html" &
    elif command -v open &> /dev/null; then
        print_status "Opening coverage report in browser..."
        open "$OUTPUT_DIR/index.html" &
    fi
fi

print_success "Test execution completed!"
