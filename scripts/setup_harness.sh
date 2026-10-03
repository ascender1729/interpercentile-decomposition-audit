#!/usr/bin/env bash
# Fetch the upstream Laban et al. harness at the commit used for the frozen corpus,
# then apply our overlay (Bedrock adapter, model_openai switch, instruction subsets).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HERE/harness}"
COMMIT=c865793fe34a929d316119b0451d01bd9183bcfd

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/microsoft/lost_in_conversation.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
cp "$HERE/harness_overlay/model_bedrock.py" "$HERE/harness_overlay/model_openai.py" "$DEST/"
cp "$HERE/harness_overlay/data/"*.json "$DEST/data/"
echo "harness ready at $DEST (upstream $COMMIT + overlay)"
