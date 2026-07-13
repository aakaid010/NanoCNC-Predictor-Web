# Deploy to Render (free tier) — single service, no extras

Your repo already has everything wired up:

- `render.yaml` — tells Render how to build & run a Python web service from `backend/`.
- `backend/app.py` — serves the Flask API **and** the static frontend (`frontend/index.html`, `style.css`, `app.js`) on the same origin, so the UI just hits `/api/predict` with no `API_BASE` config needed.
- `backend/requirements.txt` — Flask, scikit-learn, numpy, pandas, joblib, gunicorn.
- `backend/models/nanocnc_model_bundle.pkl` — already committed (≈2.3 MB), so Render ships it to the build container automatically.

You deploy **one** free web service and you're done.

---

## 1. Push the repo to GitHub (if it isn't already)

> **Heads up on Windows PowerShell:** PowerShell does **not** accept `&&` as a
> statement separator (that's bash). Use `;` instead, or run the commands on
> separate lines. The examples below use the PowerShell-friendly form.

```powershell
cd E:\thesis-mecha\nanocnc-predictor
git add -A
git commit -m "Ready for Render deploy"   # only if you have changes
git push origin main
```

If you're in **Git Bash** or **WSL**, the `&&` form is fine:

```bash
cd /e/thesis-mecha/nanocnc-predictor
git add -A && git commit -m "Ready for Render deploy" && git push origin main
```

> If the repo is private that's fine — Render can connect to private repos.

## 2. Create the Render service

1. Go to <https://dashboard.render.com/> → **New +** → **Blueprint**.
2. Connect your GitHub account if asked, then pick the repo `NanoCNC-Predictor-Web`.
3. Render reads `render.yaml` automatically. You'll see one service called `nanocnc-predictor` (Free plan).
4. Click **Apply**. That's it.

Render will:

- Set Python 3.11.9.
- Run `pip install -r requirements.txt`.
- Run `python scripts/check_bundle.py` (just verifies the bundle loads — harmless if it fails; the service still starts in Demo Mode).
- Start `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`.

First build takes ~3–5 min because of scikit-learn. After that it's fast.

## 3. Open the app

Your URL will look like `https://nanocnc-predictor.onrender.com`. Open it in a browser — you'll get the UI.

## 4. Verify it works

Edit `verify_deploy.ps1` and set `$URL` to your Render URL, then:

```powershell
cd E:\thesis-mecha\nanocnc-predictor
.\verify_deploy.ps1
```

You should see `demo_mode: false` and a successful `POST /api/predict`. If `demo_mode` is `true`, hit `/api/diag` (also in the script) to see why the bundle didn't load — usually it's a sklearn/numpy version mismatch.

> Note: free-tier services spin down after 15 min of no traffic. The first request after that takes ~30–60 s to wake up. Subsequent requests are fast.

## 5. (Optional) Upload a different model later

The UI has an **Upload Trained Model (.pkl)** section that posts to `/api/upload-model`. The file is written into `backend/models/nanocnc_model_bundle.pkl` on Render's ephemeral disk and reloaded immediately. **It will reset on the next deploy/restart**, so for a permanent swap just commit a new `nanocnc_model_bundle.pkl` and push — Render redeploys automatically.

---

## If something fails

- **Build fails on `check_bundle.py`**: that's a warning, not fatal. The service still starts in Demo Mode. To get the real model loading, look at the Render logs for the actual import error — usually the `nanocnc_model_bundle.pkl` was saved with a different numpy/sklearn than `requirements.txt` pins. Re-export with the pinned versions, or relax the pins (less safe).
- **Service returns 502**: open Render → Logs. The most common cause on the free tier is OOM during sklearn load; if that happens, we can move to a paid plan or trim the bundle.
- **`.pkl` too big to push to GitHub**: if it ever exceeds 100 MB, move it to a `nanocnc_model_bundle.pkl` download from a public URL and add a tiny `download.py` step to `buildCommand`. Not needed today (2.3 MB is fine).
- **PowerShell complains about `&&`**: that's the Windows shell, not bash. Replace every `&&` with `;` (or just put each command on its own line).

That's the whole deployment.