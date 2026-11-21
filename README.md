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

docker compose up -d server                        # STT + speaker ID on :8765
docker compose run --rm --entrypoint python edge scripts/make_conversation.py recordings/conv3.wav
docker compose run --rm edge edge --server ws://server:8765 --wav recordings/conv3.wav
docker compose run --rm edge                       # laptop mic as the "Pi"
docker compose logs -f server                      # transcript with speaker labels
```

Enroll a known voice: `docker compose run --rm server enroll Folke recordings/folke1.wav recordings/folke2.wav`.

## On the Pi

```sh
bash scripts/pi_check.sh     # hardware report + echo test; paste the output back
```
