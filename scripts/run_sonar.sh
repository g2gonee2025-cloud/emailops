#!/bin/bash
set -e

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if SONAR_TOKEN is set
if [ -z "$SONAR_TOKEN" ]; then
    echo "Error: SONAR_TOKEN environment variable is not set."
    echo "Usage: SONAR_TOKEN=your_token_here ./scripts/run_sonar.sh"
    exit 1
fi


# Auto-install SonarScanner if not present
SONAR_DIR=".sonar"
SCANNER_Version="8.0.1.6346"
SCANNER_DIR="sonar-scanner-${SCANNER_Version}-linux-x64"
SCANNER_ZIP="sonar-scanner-cli-${SCANNER_Version}-linux-x64.zip"
SCANNER_URL="https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/${SCANNER_ZIP}"

if ! command -v sonar-scanner &> /dev/null; then
    if [ ! -f "$SONAR_DIR/$SCANNER_DIR/bin/sonar-scanner" ]; then
        echo "SonarScanner not found. Downloading..."
        mkdir -p "$SONAR_DIR"
        curl -sSLo "$SONAR_DIR/$SCANNER_ZIP" "$SCANNER_URL"
        unzip -q "$SONAR_DIR/$SCANNER_ZIP" -d "$SONAR_DIR"
        rm "$SONAR_DIR/$SCANNER_ZIP"
        echo "SonarScanner installed to $SONAR_DIR/$SCANNER_DIR"
    fi
    export PATH="$PWD/$SONAR_DIR/$SCANNER_DIR/bin:$PATH"
fi

echo "Running SonarQube Scanner..."
sonar-scanner \
  -Dsonar.token="$SONAR_TOKEN" \
  -Dsonar.host.url="http://138.197.242.110:9000" \
  "$@"
