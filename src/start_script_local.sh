#!/usr/bin/env bash
set -euo pipefail

# Always run the patched startup script baked into this image.
cp /opt/comfyui-wan/start.sh /start.sh
chmod +x /start.sh

exec bash /start.sh
