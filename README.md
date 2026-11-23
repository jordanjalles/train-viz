# train-viz

A small live dashboard for training runs, built to see what a model is doing while it trains, including on a phone. Runs, loss and test-error charts, reconstructions over training, latent-noise robustness, covariance maps, FID/KID charts, and a 3-D PCA "constellation" of latent tokens you can fly through.

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `server.js`, `lib/`: one Node process, no runtime dependencies
- `public/`: the app, plus report pages from experiments
- `experiments.json`: the experiment registry (question, runs and verdict for each experiment)
- `notebook/`: cells and pages for streaming notebook runs into the dashboard

## Running it

Copy `config.example.json` to `config.json` and point `roots` at your run folders (each run: `metrics.jsonl`, `recon_<step>.png`, optional `z_<step>.pt`). Then `node server.js` and open http://127.0.0.1:8600. Tests: `node --test test/*.test.js`.

**Code:** built by Claude (Anthropic) as a coding agent, to my specification of what I needed to see.

**Requirements:** Node 20+.
