#!/bin/bash
set -e

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Auto-install SonarScanner if not present
SONAR_DIR=".sonar"
SCANNER_Version="6.2.1.4610"  # Using a known stable version with JRE often included
SCANNER_DIR="sonar-scanner-${SCANNER_Version}-linux-x64"
SCANNER_ZIP="sonar-scanner-cli-${SCANNER_Version}-linux-x64.zip"
SCANNER_URL="https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/${SCANNER_ZIP}"

# Always setup local scanner if missing
if [ ! -f "$SONAR_DIR/$SCANNER_DIR/bin/sonar-scanner" ]; then
    echo "Downloading SonarScanner..."
    mkdir -p "$SONAR_DIR"
    curl -sSLo "$SONAR_DIR/$SCANNER_ZIP" "$SCANNER_URL"
    unzip -q "$SONAR_DIR/$SCANNER_ZIP" -d "$SONAR_DIR"
    rm "$SONAR_DIR/$SCANNER_ZIP"
    echo "SonarScanner installed to $SONAR_DIR/$SCANNER_DIR"
fi

# Use the LOCAL scanner
SCANNER_BIN="$PWD/$SONAR_DIR/$SCANNER_DIR/bin/sonar-scanner"

echo "Running SonarQube Scanner..."

# Build arguments
ARGS=("-Dsonar.host.url=http://127.0.0.1:9000")

if [ -n "$SONAR_TOKEN" ]; then
    # Use sonar.token for tokens (SQ 9.9+)
    ARGS+=("-Dsonar.token=$SONAR_TOKEN")
elif [ -n "$SONAR_LOGIN" ] && [ -n "$SONAR_PASSWORD" ]; then
    ARGS+=("-Dsonar.login=$SONAR_LOGIN" "-Dsonar.password=$SONAR_PASSWORD")
else
    echo "Error: Neither SONAR_TOKEN nor (SONAR_LOGIN and SONAR_PASSWORD) are set."
    exit 1
fi

"$SCANNER_BIN" "${ARGS[@]}" "$@"
