#!/bin/bash
# Download GGUF quantized KaLM-Embedding-Gemma3-12B model for CPU inference
#
# Usage: ./scripts/download_gguf.sh [output_dir]
#
# Default output: ./models/

set -e
set -o pipefail

OUTPUT_DIR="${1:-./models}"
MODEL_REPO="mradermacher/KaLM-Embedding-Gemma3-12B-2511-GGUF"
QUANT_TYPE="Q8_0"
COMMIT_HASH="a14e6f75e83dd506f6362782e82cfbef5402b382"
CHECKSUM="546d4223bb1e940755563945f28fa12eb2fee2bc0597d3ab07638b2f5cf8030d"

echo "=== GGUF Model Downloader (8-bit) ==="
echo "Model: ${MODEL_REPO}"
echo "Quantization: ${QUANT_TYPE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found"
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

echo "Downloading GGUF model (~12GB)..."
echo "This may take 10-20 minutes depending on your connection."
echo ""

huggingface-cli download "${MODEL_REPO}" \
    --include "*${QUANT_TYPE}.gguf" \
    --local-dir "${OUTPUT_DIR}" \
    --local-dir-use-symlinks False \
    --revision "${COMMIT_HASH}"

# Find candidate GGUF file
mapfile -t GGUF_FILES < <(find "${OUTPUT_DIR}" -name "*${QUANT_TYPE}.gguf" -type f)

# Check for a unique match
if [ "${#GGUF_FILES[@]}" -ne 1 ]; then
    echo "Error: Expected to find exactly one *${QUANT_TYPE}.gguf file, but found ${#GGUF_FILES[@]}."
    if [ "${#GGUF_FILES[@]}" -gt 1 ]; then
        echo "Matching files:"
        printf " - %s\n" "${GGUF_FILES[@]}"
    fi
    exit 1
fi
GGUF_FILE="${GGUF_FILES[0]}"

# Verify checksum
echo "Verifying checksum..."
DOWNLOAD_CHECKSUM=$(sha256sum "${GGUF_FILE}" | cut -d' ' -f1)
if [ "${DOWNLOAD_CHECKSUM}" != "${CHECKSUM}" ]; then
    echo "Error: Checksum mismatch"
    echo "Expected: ${CHECKSUM}"
    echo "Got: ${DOWNLOAD_CHECKSUM}"
    exit 1
fi

SIMPLE_NAME="${OUTPUT_DIR}/kalm-12b-q8.gguf"
if [ ! -L "${SIMPLE_NAME}" ]; then
    ln -sr "${GGUF_FILE}" "${SIMPLE_NAME}"
    echo ""
    echo "Created symlink: ${SIMPLE_NAME} -> $(basename "${GGUF_FILE}")"
fi

echo ""
echo "=== Download Complete ==="
echo "Model file: ${GGUF_FILE}"
echo "Size: $(du -h "${GGUF_FILE}" | cut -f1)"
echo ""
echo "To use, set environment variable:"
echo "  export OUTLOOKCORTEX_GGUF_MODEL_PATH=${SIMPLE_NAME}"
