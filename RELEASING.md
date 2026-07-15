# Releasing `pipnn` to PyPI

Wheels (abi3, one per platform for Python 3.9+) and an sdist are built and
published by [`.github/workflows/release.yml`](.github/workflows/release.yml) on a
version tag, using **PyPI Trusted Publishing** (no API token in the repo).

## One-time setup

1. **Create the PyPI project + trusted publisher.** On PyPI → *Your projects* →
   *Publishing* (or add a "pending publisher" before the first upload):
   - PyPI project name: `pipnn`
   - Owner / repository: `iandriver` / `pipnn-scanpy`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
2. **Create the GitHub Environment** named `pypi` (Settings → Environments →
   New environment → `pypi`). No secrets needed — the workflow uses OIDC.

(Optional: repeat both for **TestPyPI** to rehearse, pointing the publish step at
`repository-url: https://test.pypi.org/legacy/`.)

## Cut a release

1. Bump the version in **`Cargo.toml`** (`[package].version`) — the single source
   of truth; `pyproject.toml` reads it via `dynamic = ["version"]`. Optionally
   bump `crates/pipnn-core/Cargo.toml` too for consistency.
2. Commit and merge to `main`.
3. Tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
   The workflow builds Linux (x86_64, aarch64) + macOS (arm64, x86_64) + Windows
   wheels and the sdist, then publishes them to PyPI.

Installs with `uv pip install pipnn` (or `pip install pipnn`) once published.

## Dry run

Trigger the workflow manually (Actions → *Release to PyPI* → *Run workflow*) — it
builds all artifacts but the publish step is gated on a tag, so nothing uploads.

## Build locally

```bash
maturin build --release --out dist   # wheel for the current platform
maturin sdist  --out dist            # source distribution
```
