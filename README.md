# Kiki

A table-top voice agent that follows a group conversation, knows who is talking, and
joins in when that makes sense. Fully self-hosted: the Pi handles audio, voice
detection and turn detection, and a local server (this laptop for now) runs
speech-to-text, speaker ID, the LLM and TTS. See [PLAN.md](PLAN.md).

## Dev quickstart (laptop, everything in Docker)

```sh
docker compose build
docker compose run --rm server fetch-models        # ~1 GB of models into ./models
docker compose run --rm --entrypoint pytest server # tests

docker compose up -d server                        # STT, speaker ID, brain, TTS on :8765 (+ llm)
docker compose run --rm --entrypoint python edge scripts/make_conversation.py recordings/conv3.wav
docker compose run --rm edge edge --server ws://server:8765 --wav recordings/conv3.wav
docker compose run --rm edge                       # laptop mic as the "Pi"
docker compose logs -f server                      # transcript with speaker labels
```

Enroll a known voice: `docker compose run --rm server enroll Folke recordings/folke1.wav recordings/folke2.wav`.

Replay with Kiki's replies saved instead of played:
`docker compose run --rm edge edge --server ws://server:8765 --wav recordings/conv3.wav --no-playback --save-replies recordings/replies`

## Diagnostics

The server serves a diagnostics page at <http://localhost:8766/>. It shows the Pi's
mic level, clipping, dropped frames, temperature and throttling every 5 s; recent
speech segments with transcript, speaker and a playable WAV; turn decisions with
the gate's probabilities; and Kiki's replies with their latency. JSON is under
`/api/status`, `/api/segments`, `/api/turns`, `/api/replies` and `/api/stats`. The
page serves room recordings, so it's bound to localhost unless you set
`KIKI_DEBUG_BIND=0.0.0.0` in `.env`.

## On the Pi (edge)

```sh
bash scripts/pi_check.sh                                   # hardware report + echo test
echo "KIKI_SERVER=ws://<laptop-ip>:8765" > .env
docker compose -f compose.pi.yaml build
docker compose -f compose.pi.yaml run --rm edge fetch-models --edge
docker compose -f compose.pi.yaml run --rm edge bench      # VAD / Smart Turn timing
docker compose -f compose.pi.yaml up -d
docker compose -f compose.pi.yaml logs -f
```
