
#!/bin/bash

set -e

REPO="cosdata/cosdata"  # Replace with your actual repo
TMP_DIR=$(mktemp -d)
ARTIFACT_NAME="cosdata"

echo "Fetching latest release..."

LATEST_RELEASE=$(curl -s "https://api.github.com/repos/$REPO/releases" | 
  grep '"tag_name":' |  
  sed -E 's/.*"([^"]+)".*/\1/' |  
  sort -V |  
  tail -n 1)

if [ -z "$LATEST_RELEASE" ]; then
    echo "Error: Could not fetch latest release."
    exit 1
fi

INSTALL_DIR="$HOME/cosdata-$LATEST_RELEASE"
BIN_DIR="$INSTALL_DIR/bin"
CONFIG_DIR="$INSTALL_DIR/config"
DATA_DIR="$INSTALL_DIR/data"

echo "Creating install directories at $INSTALL_DIR..."
mkdir -p "$BIN_DIR" "$CONFIG_DIR" "$DATA_DIR"

echo "Downloading release artifacts..."
curl -sL "https://github.com/$REPO/releases/download/$LATEST_RELEASE/$ARTIFACT_NAME" -o "$BIN_DIR/$ARTIFACT_NAME"
curl -sL "https://github.com/$REPO/releases/download/$LATEST_RELEASE/config.toml" -o "$CONFIG_DIR/config.toml"

echo "Setting executable permissions..."
chmod +x "$BIN_DIR/$ARTIFACT_NAME"

# Add the versioned install directory to PATH
if ! echo "$PATH" | grep -q "$BIN_DIR"; then
    echo "Adding $BIN_DIR to PATH..."
    echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
    echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.zshrc"
fi

# Set COSDATA_HOME for the installed version
echo "export COSDATA_HOME=\"$INSTALL_DIR\"" >> "$HOME/.bashrc"
echo "export COSDATA_HOME=\"$INSTALL_DIR\"" >> "$HOME/.zshrc"

# Create a wrapper script to prompt for the admin key on first run
LAUNCH_SCRIPT="$BIN_DIR/start-cosdata"
cat <<EOF > "$LAUNCH_SCRIPT"
#!/bin/bash
read -s -p "Enter Admin Key: " COSDATA_ADMIN_KEY
echo ""
$BIN_DIR/$ARTIFACT_NAME --admin-key "\$COSDATA_ADMIN_KEY"
export COSDATA_ADMIN_KEY=""
EOF

chmod +x "$LAUNCH_SCRIPT"

echo "Installation complete! Restart your terminal or run 'source ~/.bashrc'."
echo "Run 'start-cosdata' to start Cosdata."
