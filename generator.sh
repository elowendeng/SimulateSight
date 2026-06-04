#!/bin/bash

# ==================== Configuration ====================
# Color output settings
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==================== Function Definitions ====================
print_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_important() {
    echo -e "${RED}⚠ $1${NC}"
}

# Check if previous command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        print_success "$1 completed"
        return 0
    else
        print_error "$1 failed"
        return 1
    fi
}

# Check if directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        print_warning "Directory $1 does not exist, skipping..."
        return 1
    fi
    return 0
}

# ==================== Configuration Parameters ====================
# Data generation parameters
SUNGLASS_MODE=${SUNGLASS_MODE:-"random"}
COLORBLIND_TYPE=${COLORBLIND_TYPE:-"deuteranopia"}
EDGE_MODE=${EDGE_MODE:-"soft"}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-"png"}
SEED=${SEED:-42}
COLORBLIND_SEVERITY=${COLORBLIND_SEVERITY:-1.0}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sunglass-mode)
            SUNGLASS_MODE="$2"
            shift 2
            ;;
        --colorblind-type)
            COLORBLIND_TYPE="$2"
            shift 2
            ;;
        --edge-mode)
            EDGE_MODE="$2"
            shift 2
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --colorblind-severity)
            COLORBLIND_SEVERITY="$2"
            shift 2
            ;;
        --force-reprocess)
            FORCE_REPROCESS="--force-reprocess"
            shift
            ;;
        --no-progress)
            NO_PROGRESS="--no-progress"
            shift
            ;;
        --no-backup)
            NO_BACKUP="--no-backup"
            shift
            ;;
        --yes)
            AUTO_YES=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sunglass-mode MODE      Sunglass mode (brown/blue/strong_glare/vintage/random)"
            echo "  --colorblind-type TYPE    Colorblind type (protanopia/deuteranopia/tritanopia/random)"
            echo "  --edge-mode MODE          Edge mode (soft/sharp/binary)"
            echo "  --output-format FORMAT    Output format (png/jpg)"
            echo "  --seed SEED               Random seed"
            echo "  --colorblind-severity VAL Colorblind severity (0.0-1.0)"
            echo "  --force-reprocess         Force reprocess all images"
            echo "  --no-progress             Disable progress bar"
            echo "  --no-backup               Do not backup existing output"
            echo "  --yes                     Auto-confirm prompts"
            echo "  --help                    Show this help message"
            echo ""
            echo "Note: Generated data will be saved in the original dataset directory:"
            echo "  - Train: ./data/Train/Imgs_SL, Imgs_CB, GT_Edge"
            echo "  - Val: ./data/Val/Imgs_SL, Imgs_CB, GT_Edge"
            echo "  - Test: ./data/Test/*/Imgs_SL, Imgs_CB (GT_Edge is skipped for test)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==================== Check and Fix generator.py ====================
print_section "Checking generator.py for issues"

GENERATOR_FILE="./generator.py"

if [ -f "$GENERATOR_FILE" ]; then
    # Check for missing import time
    if ! grep -q "^import time" "$GENERATOR_FILE" && grep -q "time\.sleep" "$GENERATOR_FILE"; then
        print_warning "Adding missing 'import time' to generator.py"
        cp "$GENERATOR_FILE" "${GENERATOR_FILE}.bak"
        sed -i '1iimport time\n' "$GENERATOR_FILE"
        print_success "Fixed generator.py"
    else
        print_success "generator.py OK"
    fi
fi

# ==================== Main Pipeline ====================
print_section "Data Generation Pipeline"

print_important "Generated data will be saved in the original dataset directories!"
echo "The following folders will be created:"
echo "  - Imgs_SL (sunglass images)"
echo "  - Imgs_CB (colorblind images)"
echo "  - GT_Edge (edge maps) - ONLY for Train and Val datasets"
echo ""
echo "Note: Test datasets will NOT generate GT_Edge (not needed for testing)"
echo ""

if [ -z "$AUTO_YES" ] && [ -z "$FORCE_REPROCESS" ]; then
    read -p "Continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Data generation aborted"
        exit 1
    fi
fi

# Record start time
START_TIME=$(date +%s)
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Generation Parameters:"
echo "  - Sunglass Mode: $SUNGLASS_MODE"
echo "  - Colorblind Type: $COLORBLIND_TYPE"
echo "  - Colorblind Severity: $COLORBLIND_SEVERITY"
echo "  - Edge Mode: $EDGE_MODE"
echo "  - Output Format: $OUTPUT_FORMAT"
echo "  - Random Seed: $SEED"
echo "  - Output Location: Same as input directory"
echo ""

# Build common arguments
COMMON_ARGS="--sunglass-mode $SUNGLASS_MODE --colorblind-type $COLORBLIND_TYPE --edge-mode $EDGE_MODE --output-format $OUTPUT_FORMAT --seed $SEED --colorblind-severity $COLORBLIND_SEVERITY"

if [ -n "$FORCE_REPROCESS" ]; then
    COMMON_ARGS="$COMMON_ARGS $FORCE_REPROCESS"
fi
if [ -n "$NO_PROGRESS" ]; then
    COMMON_ARGS="$COMMON_ARGS $NO_PROGRESS"
fi
if [ -n "$NO_BACKUP" ]; then
    COMMON_ARGS="$COMMON_ARGS $NO_BACKUP"
fi

# ==================== 1. Generate Training Data ====================
print_section "Step 1/4: Generate Training Data (Train)"
if check_directory "./data/Train/Imgs"; then
    print_info "Executing: python ./data/generator.py -p ./data/Train $COMMON_ARGS"
    python ./generator.py -p ./data/Train $COMMON_ARGS
    if ! check_status "Training data generation"; then
        print_error "Training data generation failed"
        exit 1
    fi
    
    print_success "Output saved to: ./data/Train/"
    echo "  - Sunglass: ./data/Train/Imgs_SL/"
    echo "  - Colorblind: ./data/Train/Imgs_CB/"
    echo "  - Edge: ./data/Train/GT_Edge/"
else
    print_warning "Training data directory not found, skipping..."
fi

# ==================== 2. Generate Validation Data ====================
print_section "Step 2/4: Generate Validation Data (Val)"
if check_directory "./data/Val/Imgs"; then
    print_info "Executing: python ./data/generator.py -p ./data/Val $COMMON_ARGS"
    python ./generator.py -p ./data/Val $COMMON_ARGS
    check_status "Validation data generation" || true
else
    print_warning "Validation data directory not found, skipping..."
fi

# ==================== 3. Generate Test Data ====================
print_section "Step 3/4: Generate Test Data (Edge GT skipped for test sets)"

TEST_ARGS="$COMMON_ARGS --skip-edge"

# CAMO
if check_directory "./data/Test/CAMO/Imgs"; then
    print_info "Generating CAMO test data"
    python ./generator.py -p ./data/Test/CAMO $TEST_ARGS
    check_status "CAMO data generation" || true
else
    print_warning "CAMO test data directory not found, skipping..."
fi

# CHAMELEON
if check_directory "./data/Test/CHAMELEON/Imgs"; then
    print_info "Generating CHAMELEON test data"
    python ./generator.py -p ./data/Test/CHAMELEON $TEST_ARGS
    check_status "CHAMELEON data generation" || true
else
    print_warning "CHAMELEON test data directory not found, skipping..."
fi

# COD10K
if check_directory "./data/Test/COD10K/Imgs"; then
    print_info "Generating COD10K test data"
    python ./generator.py -p ./data/Test/COD10K $TEST_ARGS
    check_status "COD10K data generation" || true
else
    print_warning "COD10K test data directory not found, skipping..."
fi

# NC4K
if check_directory "./data/Test/NC4K/Imgs"; then
    print_info "Generating NC4K test data"
    python ./generator.py -p ./data/Test/NC4K $TEST_ARGS
    check_status "NC4K data generation" || true
else
    print_warning "NC4K test data directory not found, skipping..."
fi

# ==================== Completion ====================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

print_section "Data Generation Completed!"
echo -e "${GREEN}Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo -e "${GREEN}All data generation completed successfully!${NC}"
echo ""
echo "Generated data locations:"
echo "  - Training: ./data/Train/Imgs_SL, Imgs_CB, GT_Edge"
echo "  - Validation: ./data/Val/Imgs_SL, Imgs_CB, GT_Edge"
echo "  - Test: ./data/Test/*/Imgs_SL, Imgs_CB (no GT_Edge)"
