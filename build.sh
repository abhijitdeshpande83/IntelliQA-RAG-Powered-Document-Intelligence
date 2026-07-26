#!/usr/bin/env bash

set -e

# Paths
SSD_PROJECT="/Volumes/LaCie/Projects_portfolio/IntelliQA"
LOCAL_PROJECT="/Users/abhijitdeshpande/Desktop/IntelliQA"
SSD_DIST="/Volumes/LaCie/Projects_portfolio/IntelliQA/dist"
PORTFOLIO_LIBS="/Users/abhijitdeshpande/Documents/Portfolio/portfolio/libs"

log() {
    printf "\n==> %s\n" "$1"
}

log "Syncing project from SSD to local"

rsync -av --progress \
    "$SSD_PROJECT/" \
    "$LOCAL_PROJECT/"

log "Switching to local project"

cd "$LOCAL_PROJECT"

log "Cleaning previous build artifacts"

rm -rf build dist *.egg-info

log "Installing build dependencies"

python3 -m pip install --upgrade build setuptools wheel

log "Building Python package"

python3 -m build

log "Copying build artifacts to SSD"

rsync -av --progress \
    "$LOCAL_PROJECT/dist/" \
    "$SSD_DIST/"

log "Copying wheel to portfolio libraries"

cp "$SSD_DIST"/*.whl \
    "$PORTFOLIO_LIBS/"

log "Build completed successfully"

echo "Artifacts:"
echo "  • SSD      : $SSD_DIST"
echo "  • Portfolio: $PORTFOLIO_LIBS"