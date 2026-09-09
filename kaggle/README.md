# Kaggle burst training

Runs the training loop on a Kaggle GPU notebook, resuming from and writing
back to a Kaggle Dataset so sessions chain across the 12h cap.

## Read this first: Kaggle is probably slower than local

Measured on the live local run before building this:

| | this box | Kaggle GPU notebook |
|---|---|---|
| CPUs | **12** | **4 vCPU** |
| RAM | 14 GB | ~13-16 GB |
| GPU | GTX 1660 | P100 / T4 |
| **GPU utilisation** | **6-7%** | would also be ~6-7% |
| session limit | none | **12h**, ~30h/week quota |

The bottleneck is **CPU, not GPU**: 10 Node subprocesses running the
OpenFront simulation consume ~2.7 cores while the GPU sits at 6-7%. Kaggle
trades a better GPU (which this workload does not use) for a third of the
CPUs (which it does). Expect **slower** wall-clock progress per update.

`NUM_ENVS` in the notebook is therefore **3**, not the local 10 — ten workers
contending over 4 vCPU makes every `VecEnv.step()` block on the slowest one.
That also means results are **not comparable to the local run at equal update
count**: fewer envs is less diverse rollout data per update.

Use this when you want the local machine back, unattended long runs, or
parallel experiments — not for speed.

**The notebook runs on CPU (`--device cpu`), not GPU.** Confirmed on the
first real training attempt: it resumed cleanly, ran 4 updates, then crashed
with `cudaErrorNoKernelImageForDevice` — Kaggle's default PyTorch image
doesn't ship CUDA kernels compiled for the P100's compute capability
(Pascal, sm_60). Since GPU utilisation was 6-7% locally anyway, forcing CPU
costs nothing real. `enable_gpu` stays on in `kernel-metadata.json` regardless
— the accelerator tier's extra CPU/RAM allocation is worth having even with
CUDA unused.

## Prerequisite: credentials (currently missing)

The Kaggle API needs `~/.kaggle/kaggle.json`:

```json
{"username": "<your-kaggle-username>", "key": "<api-key>"}
```

Get it from kaggle.com → Account → *Create New API Token*, then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

At time of writing this box has only `~/.kaggle/access_token` (a 37-char bare
string), which is **not** what the API accepts, and no `kaggle` CLI. Upload
cannot proceed until the above exists.

## Why a Dataset instead of `git clone`

The notebook clones **OpenFrontIO** from public upstream at a pinned SHA, so
none of its 555 MB (including 156 MB of map assets) needs uploading.

It does **not** clone `openfront-rl`. When this was written the entire
multi-opponent architecture, the winner-detection fix, `recycle_every` and
`oom_guard.sh` were uncommitted working-tree changes — cloning would silently
fetch the *old, broken* code, the version where the agent was punished for
winning. So our ~0.2 MB of source ships in the Dataset instead.

**If this work gets committed and pushed, switch the notebook to clone
`openfront-rl` too and retire `package_source.sh`.**

The pinned OpenFrontIO SHA matters beyond reproducibility: the engine *is*
the simulator, so a different revision means a different environment than the
checkpoint was trained against.

## Workflow

**1. Build the payload** (~3 MB: source + checkpoint + logs):

```bash
./kaggle/package_source.sh /tmp/kgl
```

`.ntfy_topic` is excluded by default — anyone with the topic can read and
publish to that channel, and Datasets are one checkbox from public. Opt in
with `INCLUDE_NTFY=1` and keep the Dataset private.

**2. Create the Dataset** (first time):

```bash
# edit /tmp/kgl/payload/dataset-metadata.json -> replace REPLACE_USERNAME
kaggle datasets create -p /tmp/kgl/payload
```

**3. Create the notebook** on kaggle.com from
`kaggle/openfront_rl_kaggle.ipynb`, then in the sidebar:

- **Add Input** → the `openfront-rl-src` Dataset
- **Accelerator** → GPU (any; it will idle regardless)
- **Internet** → **On** (required: clones OpenFrontIO, runs `npm ci`)
- Save Version → *Save & Run All (Commit)* for the full 12h budget

**4. Chain the next session.** Download the finished notebook's `out/`
directory and push it as a new Dataset version:

```bash
kaggle datasets version -p out/ -m "session 2"
```

The next run resumes from the `latest.pt` inside it.

## Files

- **`package_source.sh`** — builds the Dataset payload from the working tree.
- **`setup_kaggle.sh`** — in-session bootstrap: clones OpenFrontIO at the
  pinned SHA, reconstructs the directory layout, `npm ci`, runs the smoke
  test. Copied into the payload so the notebook can call it.
- **`openfront_rl_kaggle.ipynb`** — the notebook.

## Notes and gotchas

- **Directory layout is load-bearing.** `env-bridge/package.json` runs
  `tsx --tsconfig ../OpenFrontIO/tsconfig.json`, so `OpenFrontIO/` must be a
  *sibling* of `env-bridge/`. `setup_kaggle.sh` reconstructs exactly that.
- **Node ≥ 18** is required by `tsx`. Kaggle ships Node for JupyterLab but
  the version has moved around; the setup script checks and fails loudly
  with the `conda install` fix rather than dying inside a tsx stack trace.
- **The smoke test runs before training.** If the engine, tsconfig paths or
  map assets are wrong it fails in seconds rather than an hour into a run.
- **Stopping is safe.** The notebook SIGTERMs training at 11h (not 12h) so
  the checkpoint write is not interrupted; a stop costs at most
  `--checkpoint-every` updates, which is the same resume-safe design the
  local run relies on.
- **Replays are not carried forward** — hundreds of MB, needed only for
  replay analysis, not for resuming. `train_log.csv` and `train_stdout.log`
  *are* carried, so the `analysis/*/territory_moving_average/` scripts keep
  working across sessions.
- **`oom_guard.sh` is not used on Kaggle.** It exists because this box runs
  VS Code and dev servers alongside training; a Kaggle session has none of
  that competing for RAM.
