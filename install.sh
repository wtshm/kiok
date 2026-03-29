#!/bin/sh
set -eu

REPO="wtshm/kiok"
INSTALL_DIR="$HOME/.local/bin"

OS="$(uname -s)"
ARCH="$(uname -m)"

# Detect target triple
case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64)  TARGET="aarch64-apple-darwin" ;;
      x86_64) TARGET="x86_64-apple-darwin" ;;
      *)
        echo "Error: unsupported architecture on $OS: $ARCH" >&2
        exit 1
        ;;
    esac
    ;;
  Linux)
    case "$ARCH" in
      x86_64) TARGET="x86_64-unknown-linux-gnu" ;;
      *)
        echo "Error: unsupported architecture on $OS: $ARCH" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Error: unsupported OS: $OS" >&2
    exit 1
    ;;
esac

# Fetch latest release tag
LATEST="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"
if [ -z "$LATEST" ]; then
  echo "Error: failed to fetch latest release from ${REPO}." >&2
  exit 1
fi

ASSET="kiok-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"

echo "Installing kiok ${LATEST} (${TARGET})..."

# Download and extract
TMPDIR_DL="$(mktemp -d)"
trap 'rm -r "$TMPDIR_DL"' EXIT

curl -fsSL "$URL" -o "${TMPDIR_DL}/${ASSET}"
tar xzf "${TMPDIR_DL}/${ASSET}" -C "$TMPDIR_DL"

# Install binary
mkdir -p "$INSTALL_DIR"
mv "${TMPDIR_DL}/kiok" "${INSTALL_DIR}/kiok"
chmod +x "${INSTALL_DIR}/kiok"

echo "Installed kiok to ${INSTALL_DIR}/kiok"

# Check PATH
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    echo ""
    echo "Add ${INSTALL_DIR} to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    echo ""
    ;;
esac

# Install Claude Code skill
if command -v npx >/dev/null 2>&1; then
  echo "Installing recall-kiok skill..."
  npx skills install "${REPO}"
else
  echo "npx not found — skip skill install. Run manually:"
  echo "  npx skills install ${REPO}"
fi

echo ""
echo "Run 'kiok setup' to complete setup."
