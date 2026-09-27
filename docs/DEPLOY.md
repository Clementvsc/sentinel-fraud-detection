# Deploying Sentinel publicly

Sentinel is a **stateful FastAPI app** with a background traffic simulator and a
WebSocket feed. Pick a host that can run a long‑lived process:

| Host | Free? | Fits? | Effort |
|---|---|---|---|
| **Hugging Face Spaces** (Docker) | ✅ forever | ✅ full app, WebSocket, sleeps when idle | 2 min |
| **Render** (Docker web service) | ✅ (spins down after 15 min idle) | ✅ full app, WebSocket | 2 min |
| **Fly.io** | ✅ small allowance | ✅ full app, WebSocket | 3 min |
| **Vercel** | ✅ | ⚠️ *lite* mode only — no background loop, no WebSocket (poll‑driven) | 3 min |

A pre‑trained model (`sentinel/artifacts/model.joblib`, ~700 KB) is committed, so
every target starts instantly — no training on boot.

---

## Hugging Face Spaces  (recommended — free, permanent, ML‑friendly)

1. Create a Space → **SDK: Docker** → **blank**.
2. Push this repo to it (`git remote add space https://huggingface.co/spaces/<you>/sentinel && git push space main`).
3. Prepend this block to the top of `README.md` (Spaces requires it):

   ```yaml
   ---
   title: Sentinel Fraud Detection
   emoji: 🛡️
   colorFrom: blue
   colorTo: purple
   sdk: docker
   app_port: 8000
   pinned: false
   ---
   ```

Public URL: `https://<you>-sentinel.hf.space`.

---

## Render

1. New → **Blueprint**, point at this repo (it reads [`render.yaml`](../render.yaml)).
2. Deploy. Public URL: `https://sentinel-<hash>.onrender.com`.

Optionally add a free Render Redis instance and set `SENTINEL_REDIS_URL` so warm
behavioural state survives restarts and is shared across instances.

---

## Fly.io

```bash
fly launch --copy-config --now     # uses fly.toml
```

Public URL: `https://sentinel-fraud.fly.dev`.

---

## Vercel  (lite mode)

Vercel Python functions are request‑scoped: **no background loop, no WebSocket**.
`api/index.py` sets `SENTINEL_SERVERLESS=1`, which switches the app to
**poll mode** — the dashboard calls `POST /tick` for ambient traffic and attack
injections are scored synchronously. State is per‑instance and resets on cold
start (attach an external Redis via `SENTINEL_REDIS_URL` for continuity).

```bash
npm i -g vercel && vercel        # reads vercel.json + api/requirements.txt
```

The explainability path uses the existing classifier and adds no package dependency, keeping the function small. Counterfactual paths are hypothetical reference comparisons, not causal advice.

Public URL: `https://<project>.vercel.app`.

---

## Any Docker host

```bash
docker build -t sentinel .
docker run -p 8000:8000 sentinel
# or the 2-worker + Redis stack:
docker compose up --scale sentinel=2
```

## Note on public exposure

The demo has no auth and the simulator can be driven by anyone with the URL.
That's fine for a hackathon demo on synthetic data. Before putting anything real
behind it, add authentication and rate‑limiting, and disable the `/simulator/*`
routes.
