#!/bin/zsh
# Packages the Windows version: dist/CallTranscriber-Windows.zip
set -e
cd "$(dirname "$0")/../.."
OUT=dist/CallTranscriber-Windows.zip
STAGE=$(mktemp -d)/CallTranscriber
mkdir -p "$STAGE/server" "$STAGE/app/windows" "$STAGE/app/desktop" dist
rsync -a --exclude node_modules app/desktop/ "$STAGE/app/desktop/"
cp server/server.py server/sync.py server/voiceprint.py server/langwords.txt.gz server/pyproject.toml server/uv.lock server/.python-version "$STAGE/server/"
cp -R server/dashboard "$STAGE/server/dashboard"; find "$STAGE" -name events -type d -prune -exec rm -rf {} +
cp app/windows/call_transcriber.pyw app/windows/icon.ico app/windows/icon.png "$STAGE/app/windows/"
cp install.ps1 install.cmd "$STAGE/"
cp README-WINDOWS.md "$STAGE/README.md"
mkdir -p "$STAGE/transcripts"
rm -f "$OUT"
(cd "$(dirname "$STAGE")" && zip -qr -X "$OLDPWD/$OUT" CallTranscriber)
echo "Built $OUT ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT" | tail -n +4 | awk '{print $4}' | grep -v '/$' | sed 's|^CallTranscriber/||' | head -40
