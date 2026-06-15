# Project Page: The Gaussian Is Enough

## Preview locally

```bash
python3 -m http.server 8765 -d /home/chenxu/Downloads/anzu_personal/Videos/New_prior_paper/New_prior_website
```

Then open http://localhost:8765 in your browser. Ctrl+C to stop.

## Deploy to GitHub Pages

1. Create a new repo (e.g., `gaussian-is-enough`)
2. Push this folder's contents (index.html + Assets/) to the `main` branch
3. Go to repo Settings → Pages → Source: "Deploy from a branch" → Branch: `main` → Save
4. Site will be live at `https://<username>.github.io/gaussian-is-enough/`

## Files

- `index.html` — entire page (self-contained, all CSS inline)
- `Assets/*.png` — figures exported from the paper PDFs at 200 DPI
