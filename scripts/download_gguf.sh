#!/bin/bash
# Download GGUF quantized KaLM-Embedding-Gemma3-12B model for CPU inference
#
# Usage: ./scripts/download_gguf.sh [output_dir]
#
# Default output: ./models/

set -e

OUTPUT_DIR="${1:-./models}"
MODEL_REPO="mradermacher/KaLM-Embedding-Gemma3-12B-2511-GGUF"
QUANT_TYPE="Q8_0"

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
    --local-dir-use-symlinks False

# Find the downloaded file and create a symlink with a simple name
GGUF_FILE=$(find "${OUTPUT_DIR}" -name "*${QUANT_TYPE}.gguf" -type f | head -n1)

if [ -n "${GGUF_FILE}" ]; then
    SIMPLE_NAME="${OUTPUT_DIR}/kalm-12b-q8.gguf"
    if [ ! -e "${SIMPLE_NAME}" ]; then
        ln -s "$(basename "${GGUF_FILE}")" "${SIMPLE_NAME}"
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
else
    echo "Error: GGUF file not found after download"
    exit 1
fi
