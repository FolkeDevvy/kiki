#!/usr/bin/env bash
# Milestone 0 hardware report. Run ON THE PI:  bash scripts/pi_check.sh
# Needs only alsa-utils + python3 (both on Raspberry Pi OS). Plays ~3 s of noise
# through the USB speakerphone and records at the same time to measure echo.
set -euo pipefail

hr() { printf '\n== %s ==\n' "$1"; }

hr "board"
tr -d '\0' </proc/device-tree/model 2>/dev/null; echo
grep -E 'MemTotal' /proc/meminfo
uname -srm
. /etc/os-release && echo "$PRETTY_NAME"

hr "storage / boot"
findmnt -no SOURCE / || true
lsblk -o NAME,SIZE,TYPE,MOUNTPOINTS | grep -vE '^(loop|zram)' || true

hr "usb"
lsusb

hr "sound cards"
cat /proc/asound/cards
echo "audio server: $( (pactl info 2>/dev/null | grep 'Server Name') || echo 'none (plain ALSA)')"

CARD="${KIKI_CARD:-$(awk '/USB-Audio/ {print $1; exit}' /proc/asound/cards)}"
if [[ -z "$CARD" ]]; then
  echo "No USB audio card found; set KIKI_CARD=<card number> and re-run."; exit 1
fi
DEV="plughw:${CARD},0"
echo "using card $CARD ($DEV)"

hr "capture / playback formats"
arecord -D "hw:${CARD},0" --dump-hw-params -d 1 /dev/null 2>&1 | grep -E 'RATE|CHANNELS|FORMAT' || true
aplay   -D "hw:${CARD},0" --dump-hw-params /dev/zero -d 1 2>&1 | grep -E 'RATE|CHANNELS|FORMAT' || true

hr "echo test"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python3 - "$TMP/noise.wav" <<'EOF'
import math, random, struct, sys, wave
sr = 16000
random.seed(0)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    frames = bytearray()
    for i in range(sr * 3):  # 3 s of speech-band noise bursts, 250 ms on / 250 ms off
        on = (i // (sr // 4)) % 2 == 0
        v = random.gauss(0, 6000) if on else 0
        frames += struct.pack("<h", max(-32767, min(32767, int(v))))
    w.writeframes(bytes(frames))
EOF
echo "recording 6 s; playing noise from t=1.5 s (it will be loud-ish)"
arecord -q -D "$DEV" -f S16_LE -r 16000 -c 1 -d 6 "$TMP/rec.wav" &
REC=$!
sleep 1.5
aplay -q -D "$DEV" "$TMP/noise.wav"
wait $REC
python3 - "$TMP/rec.wav" <<'EOF'
import math, struct, sys, wave
with wave.open(sys.argv[1]) as w:
    sr = w.getframerate()
    a = struct.unpack(f"<{w.getnframes()}h", w.readframes(w.getnframes()))
def rms(x): return math.sqrt(sum(v * v for v in x) / max(1, len(x))) or 1.0
floor = rms(a[int(0.2 * sr):int(1.2 * sr)])
during = rms(a[int(1.8 * sr):int(4.3 * sr)])
db = 20 * math.log10(during / floor)
print(f"noise floor rms={floor:.0f}  during playback rms={during:.0f}  echo={db:+.1f} dB above floor")
if db < 6:
    print("=> speaker barely reaches the mic: hardware echo cancellation looks ACTIVE")
elif db < 20:
    print("=> partial suppression: some AEC, probably needs software AEC on top")
else:
    print("=> strong echo: no effective hardware AEC, we'll need PipeWire webrtc echo-cancel")
EOF

hr "docker"
docker --version 2>/dev/null || echo "docker not installed (curl -fsSL https://get.docker.com | sh)"

echo
echo "Paste everything above back to Claude."
