#!/usr/bin/env bash
# Install or update the recapture CLI as a uv tool, at the version this skill is for.
#
#   bash install.sh [--editor]
#
# --editor  also install the editor window (imgui-bundle, about 30 MB)
#
# Needs uv (https://docs.astral.sh/uv/), git and network access to github.com.
# Set RECAPTURE_SOURCE to install from somewhere else, e.g. a local clone.
# Assets are a separate step: `recapture fetch-assets`.
set -euo pipefail

VERSION="0.1.0"
SOURCE="${RECAPTURE_SOURCE:-git+https://github.com/PSPDFKit-labs/recapture.git@v${VERSION}}"
EXTRAS=""
for arg in "$@"; do
  case "$arg" in
    --editor) EXTRAS="[editor]" ;;
    *) echo "unknown option: $arg (use --editor)" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/ and run this again." >&2
  exit 3
fi

if command -v recapture >/dev/null 2>&1 && [ "$(recapture --version 2>/dev/null)" = "recapture ${VERSION}" ] && [ -z "$EXTRAS" ]; then
  echo "recapture ${VERSION} is already installed."
  exit 0
fi

echo "Installing recapture${EXTRAS} ${VERSION} from ${SOURCE}"
if ! uv tool install --force "recapture${EXTRAS} @ ${SOURCE}"; then
  echo "The install failed. Check that this machine can reach github.com over HTTPS," >&2
  echo "and that the tag v${VERSION} exists in https://github.com/PSPDFKit-labs/recapture." >&2
  exit 4
fi

if ! command -v recapture >/dev/null 2>&1; then
  echo "Installed, but recapture is not on PATH. Run: uv tool update-shell, then open a new terminal." >&2
  exit 5
fi
recapture --version
echo "Next: recapture fetch-assets (about 105 MB of CC0 textures, once per machine)."
