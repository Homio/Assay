#!/usr/bin/env bash
# Start Label Studio (isolated venv) with local-file serving of ../label/.
# Open http://localhost:8080 in a browser after it boots.
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/homio/sda/1111/label
exec /home/homio/sda/1111/label/ls-venv/bin/label-studio start --port 8080 "$@"
