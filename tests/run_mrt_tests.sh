#!/bin/bash
# ============================================================================
# MRT-CM / MRT-RM Compilation and Unit Test Runner
# ============================================================================
# Usage: bash run_mrt_tests.sh
#
# Steps:
#   1. Compile CUDA unit test in MRT-CM mode
#   2. Compile CUDA unit test in MRT-RM mode
#   3. Run MRT-CM tests on GPU
#   4. Run MRT-RM tests on GPU
#   5. Compile main.cu with USE_MRT_CM=1 to verify full project compiles
#
# Prerequisites:
#   - CUDA toolkit (nvcc) in PATH
#   - GPU with compute capability >= sm_80 (A100/A30/etc.)
#   - All project headers in place
#   - Run from the tests/ directory: cd tests && bash run_mrt_tests.sh
# ============================================================================

set -e  # Exit on first error

# Project root (one level up from tests/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ARCH="sm_35"  # Kepler architecture (your GPU)

echo "============================================"
echo "  MRT-CM / MRT-RM Test Suite"
echo "============================================"
echo ""

# ── Step 1: Compile MRT-CM unit test ──
echo -e "${YELLOW}[Step 1/5]${NC} Compiling CUDA unit test (MRT-CM mode)..."
nvcc -O2 -arch=${ARCH} test_mrt_cm_cuda.cu -o test_mrt_cm -I"${PROJECT_ROOT}" \
    -DUSE_MRT=1 -DUSE_MRT_CM=1 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ MRT-CM compilation successful${NC}"
else
    echo -e "  ${RED}✗ MRT-CM compilation FAILED${NC}"
    exit 1
fi
echo ""

# ── Step 2: Compile MRT-RM unit test ──
echo -e "${YELLOW}[Step 2/5]${NC} Compiling CUDA unit test (MRT-RM mode)..."
nvcc -O2 -arch=${ARCH} test_mrt_cm_cuda.cu -o test_mrt_rm -I"${PROJECT_ROOT}" \
    -DUSE_MRT=1 -DUSE_MRT_CM=0 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ MRT-RM compilation successful${NC}"
else
    echo -e "  ${RED}✗ MRT-RM compilation FAILED${NC}"
    exit 1
fi
echo ""

# ── Step 3: Run MRT-CM tests ──
echo -e "${YELLOW}[Step 3/5]${NC} Running MRT-CM GPU unit tests..."
./test_mrt_cm
CM_RESULT=$?
echo ""

# ── Step 4: Run MRT-RM tests ──
echo -e "${YELLOW}[Step 4/5]${NC} Running MRT-RM GPU unit tests..."
./test_mrt_rm
RM_RESULT=$?
echo ""

# ── Step 5: Compile main.cu with MRT-CM ──
echo -e "${YELLOW}[Step 5/5]${NC} Compiling main.cu with USE_MRT_CM=1 (full project)..."
# Detect MPI compiler
MPICXX=$(which mpicxx 2>/dev/null || which mpic++ 2>/dev/null || echo "")
if [ -z "$MPICXX" ]; then
    echo -e "  ${YELLOW}⚠ mpicxx not found, skipping main.cu compilation${NC}"
    echo "  (Unit tests still valid — main.cu requires MPI)"
else
    # Extract MPI flags, wrapping gcc-only flags for nvcc
    MPI_INCLUDE=$(${MPICXX} --showme:compile 2>/dev/null || echo "")
    MPI_LINK_RAW=$(${MPICXX} --showme:link 2>/dev/null || echo "")
    # Separate -L/-l flags (nvcc understands) from compiler flags like -pthread (nvcc doesn't)
    MPI_LIBS=""
    MPI_XCOMPILER=""
    for flag in ${MPI_LINK_RAW}; do
        case "$flag" in
            -L*|-l*|-Wl,*) MPI_LIBS="${MPI_LIBS} ${flag}" ;;
            *)              MPI_XCOMPILER="${MPI_XCOMPILER},${flag}" ;;
        esac
    done
    XCOMP_FLAG=""
    if [ -n "${MPI_XCOMPILER}" ]; then
        XCOMP_FLAG="-Xcompiler ${MPI_XCOMPILER#,}"
    fi

    nvcc -O2 -arch=${ARCH} "${PROJECT_ROOT}/main.cu" -o main_cm -I"${PROJECT_ROOT}" \
        -DUSE_MRT=1 -DUSE_MRT_CM=1 \
        ${MPI_INCLUDE} ${MPI_LIBS} ${XCOMP_FLAG} 2>&1
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ main.cu (MRT-CM) compilation successful${NC}"
    else
        echo -e "  ${RED}✗ main.cu (MRT-CM) compilation FAILED${NC}"
        exit 1
    fi

    # Also verify MRT-RM still compiles
    echo "  Verifying main.cu with USE_MRT_CM=0 (MRT-RM)..."
    nvcc -O2 -arch=${ARCH} "${PROJECT_ROOT}/main.cu" -o main_rm -I"${PROJECT_ROOT}" \
        -DUSE_MRT=1 -DUSE_MRT_CM=0 \
        ${MPI_INCLUDE} ${MPI_LIBS} ${XCOMP_FLAG} 2>&1
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ main.cu (MRT-RM) compilation successful${NC}"
    else
        echo -e "  ${RED}✗ main.cu (MRT-RM) compilation FAILED${NC}"
        exit 1
    fi
fi
echo ""

# ── Summary ──
echo "============================================"
echo "  SUMMARY"
echo "============================================"
if [ $CM_RESULT -eq 0 ] && [ $RM_RESULT -eq 0 ]; then
    echo -e "  ${GREEN}ALL TESTS PASSED${NC}"
    echo ""
    echo "  You can now run simulations with either mode:"
    echo "    MRT-CM: set USE_MRT_CM=1 in variables.h"
    echo "    MRT-RM: set USE_MRT_CM=0 in variables.h"
    exit 0
else
    echo -e "  ${RED}SOME TESTS FAILED${NC}"
    [ $CM_RESULT -ne 0 ] && echo -e "    ${RED}✗ MRT-CM tests failed${NC}"
    [ $RM_RESULT -ne 0 ] && echo -e "    ${RED}✗ MRT-RM tests failed${NC}"
    exit 1
fi
