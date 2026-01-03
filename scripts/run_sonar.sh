#!/bin/bash
set -e

# Load .env file if it exists
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        # Remove surrounding quotes and export only SONAR variables
        if [[ $key == SONAR_* ]]; then
            eval export "$key=\"$(echo $value | sed -e 's/^"//' -e 's/"$//')\""
        fi
    done < <(grep -v '^#' .env | grep -v '^$')
fi

# Auto-install SonarScanner if not present
SONAR_DIR=".sonar"
SCANNER_VERSION="6.2.1.4610"  # Using a known stable version with JRE often included
SCANNER_DIR="sonar-scanner-${SCANNER_VERSION}-linux-x64"
SCANNER_ZIP="sonar-scanner-cli-${SCANNER_VERSION}-linux-x64.zip"
SCANNER_URL="https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/${SCANNER_ZIP}"

# Always setup local scanner if missing
if [ ! -x "$SONAR_DIR/$SCANNER_DIR/bin/sonar-scanner" ]; then
    echo "SonarScanner not found or not executable, setting up..."

    # Create a temporary directory for secure download and extraction
    TMP_DIR=$(mktemp -d)
    # Ensure cleanup on script exit or interruption
    trap 'rm -rf "$TMP_DIR"' EXIT

    ZIP_PATH="$TMP_DIR/$SCANNER_ZIP"

    echo "Downloading SonarScanner from $SCANNER_URL..."
    # Use --fail to handle HTTP errors properly
    if ! curl --fail -sSLo "$ZIP_PATH" "$SCANNER_URL"; then
        echo "ERROR: Failed to download SonarScanner. Check URL and network." >&2
        exit 1
    fi

    # SECURITY: A checksum verification should be added here if available.
    echo "WARNING: Archive integrity is not verified (checksum unavailable)."

    echo "Unzipping archive..."
    # Extract to temp dir to prevent path traversal (zip-slip)
    if ! unzip -q "$ZIP_PATH" -d "$TMP_DIR"; then
        echo "ERROR: Failed to unzip SonarScanner archive." >&2
        exit 1
    fi

    # Move the extracted scanner to its final destination
    if [ -d "$TMP_DIR/$SCANNER_DIR" ]; then
        mkdir -p "$SONAR_DIR"
        # Ensure the destination is clean before moving
        rm -rf "$SONAR_DIR/$SCANNER_DIR"
        mv "$TMP_DIR/$SCANNER_DIR" "$SONAR_DIR/"
        echo "SonarScanner installed successfully to $SONAR_DIR/$SCANNER_DIR"
    else
        echo "ERROR: Expected directory '$SCANNER_DIR' not found in archive." >&2
        exit 1
    fi
fi

# Use the LOCAL scanner
SCANNER_BIN="$PWD/$SONAR_DIR/$SCANNER_DIR/bin/sonar-scanner"

# Pre-execution check for the scanner binary
if [ ! -x "$SCANNER_BIN" ]; then
    echo "ERROR: SonarScanner binary not found or not executable at $SCANNER_BIN" >&2
    exit 1
fi

echo "Running SonarQube Scanner..."

# Build arguments
# SECURITY: Pass credentials via environment variables, not command-line args.
# SonarScanner automatically picks up SONAR_HOST_URL, SONAR_TOKEN, etc.
export SONAR_HOST_URL="${SONAR_HOST_URL:-http://127.0.0.1:9000}"

# Check for required credentials
if [ -z "$SONAR_TOKEN" ] && { [ -z "$SONAR_LOGIN" ] || [ -z "$SONAR_PASSWORD" ]; }; then
    echo "Error: Required environment variables are not set." >&2
    echo "Please set SONAR_TOKEN, or both SONAR_LOGIN and SONAR_PASSWORD." >&2
    exit 1
fi

# Pass additional user-provided args to the scanner
"$SCANNER_BIN" "$@"
