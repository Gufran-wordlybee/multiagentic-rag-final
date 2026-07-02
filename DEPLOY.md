# Deploying MultiAgentic RAG to Render (free tier)

This guide deploys the app as a live public URL using [Render](https://render.com), using the `Dockerfile` already included in `MultiAgenticRAG/`. Render's free web service tier is enough to run this — no credit card required to start.

Total time: ~10 minutes.

---

## 1. Push the code to GitHub

Render deploys from a Git repo.

```bash
cd MultiAgenticRAG
git init                     # skip if already a git repo
git add .
git commit -m "Deployable MultiAgentic RAG"
```

Create a new repo on GitHub, then:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

**Double-check `.env` is not being committed** — it's in `.gitignore`, but run `git status` and confirm you don't see it staged. Only `.env.example` should be tracked.

---

## 2. Create a Render account

Go to [render.com](https://render.com) and sign up (GitHub sign-in is fastest — it also makes connecting your repo a one-click step).

---

## 3. Create a new Web Service

1. From the Render dashboard, click **New +** → **Web Service**.
2. Choose **Build and deploy from a Git repository**, then connect the GitHub repo you just pushed.
3. Render will scan the repo. Configure:

   | Field | Value |
   |---|---|
   | **Name** | `multiagentic-rag` (or anything you like) |
   | **Region** | closest to you |
   | **Branch** | `main` |
   | **Root Directory** | `MultiAgenticRAG` (since the Dockerfile lives inside this subfolder, not the repo root) |
   | **Runtime** | **Docker** (Render should auto-detect the Dockerfile once Root Directory is set) |
   | **Instance Type** | **Free** |

---

## 4. Set environment variables

Still on the same setup page (or under **Environment** after creation), add:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | your Groq API key |
| `COHERE_API_KEY` | your Cohere API key |

(`PORT` is set automatically by Render — the Dockerfile's `uvicorn` command respects Render's `$PORT` via the `CMD` binding to `0.0.0.0:8000`; Render's free tier maps external traffic to whatever port you `EXPOSE`, so no changes needed there.)

Optional tuning variables, only add if you want to change the defaults:

| Key | Default | What it does |
|---|---|---|
| `MAX_UPLOAD_MB` | `25` | Max PDF upload size |
| `SESSION_TTL_SECONDS` | `86400` | How long an uploaded PDF + its index stays before auto-deletion |

---

## 5. Deploy

Click **Create Web Service**. Render will:

1. Pull your repo
2. Build the Docker image (this installs Python deps, docling, and pre-downloads the embedding model — expect the **first build** to take 5–10 minutes)
3. Start the container and run the healthcheck against `/health`

You'll see build logs streaming live. Once it says **Live**, your app is up at a URL like:

```
https://multiagentic-rag.onrender.com
```

Open it, upload a PDF, and start asking questions.

---

## 6. Notes on the free tier

- **Spin-down on idle**: Render's free web services spin down after ~15 minutes of no traffic and take ~30–60 seconds to wake back up on the next request. That's expected — the first request after idle will just be slow, not broken.
- **Ephemeral disk**: the free tier's filesystem is not persistent across deploys/restarts. Uploaded PDFs and their indexes (`sessions/`) will be wiped on redeploy — that's fine for this app's per-session design (users just re-upload), but don't rely on old sessions surviving a redeploy.
- **No GPU**: docling runs on CPU here (already configured in `retriever/retriever.py` via `AcceleratorDevice.CPU`), so very large/complex PDFs will take longer to index than on a machine with GPU acceleration. For typical documents (a few dozen pages) this is a matter of seconds to low tens of seconds.

---

## Redeploying after changes

Render auto-deploys on every push to your connected branch by default:

```bash
git add .
git commit -m "some change"
git push
```

Watch the **Events** tab on the Render dashboard for build/deploy progress.

---

## Alternative hosts

The same `Dockerfile` works unmodified on any Docker-capable host:

- **Railway** — similar to Render, also has a free/trial tier, "Deploy from GitHub repo" flow.
- **Fly.io** — `fly launch` in the `MultiAgenticRAG/` directory detects the Dockerfile automatically; free allowance covers small always-on instances.
- **Any VPS** (e.g. a $4-6/mo droplet) — `docker build -t multiagentic-rag . && docker run -p 8000:8000 --env-file .env multiagentic-rag`.
