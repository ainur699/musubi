#!/usr/bin/env python3
"""Score a musubi-tuner LoRA checkpoint with the kink_oracles VLM metrics.

End-to-end CLI for one experiment (default ``_train/exp/without_vae``):

  1. Picks the latest ``*.safetensors`` checkpoint in ``<exp>/output_all`` (or
     ``--checkpoint``) and injects it into the ComfyUI pipeline's ``LoRA_1``
     placeholder via ``ComfyCheckpointGenerator`` (the other ``LoRA_*`` are
     disabled; everything else in the workflow is left as the author saved it).
  2. Generates one image per kink (``kink_oracles.load_kinks`` -> ``gandalf_prompt``)
     on the ComfyUI server, using that kink's per-row reference avatar
     (``<avatars>/<NNN>.png``) as the edit reference. Images are saved to
     ``<exp>/metrics/<checkpoint-name>/images/<NNN>_<kink>.png`` (resume-safe).
  3. Runs the kink_oracles clone-validation suite — the full kink suite
     (kink/anatomy, fine_anatomy, aesthetics x2, prompt_following) PLUS
     ``clone_quality`` (identity transfer from the avatar) — and writes the
     reports (``oracle_results.json``, ``validation_summary.json``, heatmaps) to
     ``<exp>/metrics/<checkpoint-name>/``.

Oracle model: a single Gemini model (default ``~google/gemini-pro-latest``, the
OpenRouter "latest" Gemini Pro pointer; the kink_oracles guard is patched to strip
the '~' so the served concrete id is accepted). There is NO fallback: a preflight check
errors out if the model is unavailable or if OpenRouter would serve a non-Gemini
model, and a post-run guard fails if any oracle result came from a non-Gemini
model — so the run never silently falls back to grok / another model.

Secrets come from ``--env-file`` (default ``comfy-service-pipelines/.env``):
``OPENROUTER_API_KEY`` (oracle) and ``AWS_*`` (one-time S3 avatar/reference
download). Reference data is cached under ``--cache-dir``
(default ``/mnt/images/a.gainetdionov/vision-oracles``).

Run with an environment that has ``kink_oracles`` + ``insightface`` + ``onnxruntime``
(the ``base`` conda env), e.g.:

    conda run -n base python _train/scripts/eval_oracle_metric.py --top-n 5
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("eval_oracle_metric")

# --- repo layout -----------------------------------------------------------
SCRIPT = Path(__file__).resolve()
MUSUBI_ROOT = SCRIPT.parents[2]              # .../musubi-tuner
GITHUB_ROOT = SCRIPT.parents[3]              # .../Github
COMFY_REPO = GITHUB_ROOT / "comfy-service-pipelines"
COMFY_SCRIPTS = COMFY_REPO / "scripts"

# --- defaults --------------------------------------------------------------
DEFAULT_EXP_DIR = MUSUBI_ROOT / "_train" / "exp" / "without_vae"
DEFAULT_PIPELINE = COMFY_REPO / "pipelines" / "qwen_image_ref_only_vlm_in_the_wild.json"
DEFAULT_BASE_URL = "http://10.0.8.31:8188"
DEFAULT_CACHE_DIR = "/mnt/images/a.gainetdionov/vision-oracles"
DEFAULT_AVATARS = "/mnt/images/a.gainetdionov/vision-oracles/avatar_refs"
DEFAULT_LORAS_DIR = "/mnt/images/a.gainetdionov/ComfyUI/models/loras"
DEFAULT_ENV = COMFY_REPO / ".env"
# OpenRouter's auto-updating "latest" Gemini Pro pointer (note the leading '~');
# Floating "latest" alias, matching the '[glatest]' baselines (scored with this
# exact id). The kink_oracles oracle.py guard is patched to strip the '~' when
# comparing providers, so the served concrete 'google/...' id (currently
# gemini-3.1-pro) is accepted. Plain 'google/gemini-pro-latest' (no '~') is invalid.
DEFAULT_MODEL = "~google/gemini-pro-latest"


def load_env(path: Path) -> None:
    """Minimal .env loader (no dependency); values override the environment."""
    if not path.exists():
        logger.warning(".env not found at %s (relying on existing environment)", path)
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def step_of(path: Path) -> int:
    """Training step parsed from a '...-step00016200.safetensors' name (-1 if none)."""
    match = re.search(r"step0*(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def latest_checkpoint(folder: Path) -> Path:
    checkpoints = sorted(folder.glob("*.safetensors"), key=step_of)
    if not checkpoints:
        raise SystemExit(f"No *.safetensors checkpoints in {folder}")
    return checkpoints[-1]


def check_model_available(model: str, allow_non_gemini: bool) -> None:
    """Fail fast (no fallback) if the oracle model can't be used, so the run is
    scored with the intended Gemini model and never silently substitutes grok."""
    import requests

    if not allow_non_gemini and "gemini" not in model.lower():
        raise SystemExit(
            f"--model {model!r} is not a Gemini model. This eval is Gemini-only; "
            f"pass --allow-non-gemini to override deliberately."
        )
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set (add it to the .env)")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            timeout=60,
        )
        data = response.json()
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"Oracle model {model!r} check failed: {error}")
    if response.status_code != 200 or "choices" not in data:
        message = data.get("error", {}).get("message", "") if isinstance(data, dict) else ""
        raise SystemExit(
            f"Oracle model {model!r} is unavailable (HTTP {response.status_code}): "
            f"{message or str(data)[:200]}"
        )
    served = str(data.get("model", "")) if isinstance(data, dict) else ""
    if not allow_non_gemini and served and "gemini" not in served.lower():
        raise SystemExit(
            f"OpenRouter served {served!r} for request {model!r} — refusing "
            f"(no fallback to a non-Gemini model)."
        )
    logger.info("oracle preflight OK: requested=%s served=%s", model, served or "?")


def guard_no_fallback(results: list, model: str, allow_non_gemini: bool) -> None:
    """Hard error if any oracle result came from a non-Gemini model."""
    if allow_non_gemini:
        return
    seen: set[str] = set()
    for r in results:
        for key, value in r.items():
            if key.endswith("_full") and isinstance(value, dict):
                for oracle in value.get("all_oracle_results", []) or []:
                    name = oracle.get("model")
                    if name:
                        seen.add(name)
    bad = sorted(m for m in seen if "gemini" not in m.lower())
    if bad:
        raise SystemExit(
            f"Detected non-Gemini oracle model(s) in results: {bad}. "
            f"Expected only {model!r} — failing instead of reporting a "
            f"fallback-scored metric."
        )
    logger.info("no-fallback guard OK: oracle models used = %s", sorted(seen) or [model])


def ensure_avatars(avatars_base: Path) -> Path:
    """Return a dir of per-row avatars (<NNN>.png), downloading from S3 if absent."""
    if avatars_base.exists() and any(avatars_base.glob("*.png")):
        return avatars_base
    from kink_oracles.refs import ensure_avatar_refs

    logger.info("avatars not found at %s; downloading from S3 to cache ...", avatars_base)
    resolved = ensure_avatar_refs(None)  # uses KINK_ORACLES_CACHE_DIR
    if not resolved or not any(Path(resolved).glob("*.png")):
        raise SystemExit(f"Failed to obtain reference avatars (got {resolved!r})")
    return Path(resolved)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR,
                   help="Experiment dir (holds output_all/ and metrics/)")
    p.add_argument("--checkpoints-dir", type=Path, default=None,
                   help="Folder of *.safetensors (default: <exp-dir>/output_all)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Specific checkpoint (default: latest by step)")
    p.add_argument("--metrics-dir", type=Path, default=None,
                   help="Where to write <checkpoint-name>/ (default: <exp-dir>/metrics)")
    p.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--avatars-base", type=Path, default=Path(DEFAULT_AVATARS))
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help="KINK_ORACLES_CACHE_DIR for reference/avatar data")
    p.add_argument("--loras-dir", type=Path, default=Path(DEFAULT_LORAS_DIR),
                   help="ComfyUI loras dir (must be visible to the server)")
    p.add_argument("--lora-subdir", default=None,
                   help="Subfolder under loras-dir (default: exp-dir name)")
    p.add_argument("--lora-title", default="LoRA_1")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Gemini oracle model")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--top-n", type=int, default=-1, help="Limit to first N kinks (-1 = all)")
    p.add_argument("--n-jobs", type=int, default=64, help="Parallel oracle calls")
    p.add_argument("--timeout", type=int, default=300, help="Per-generation timeout (s)")
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--comfy-scripts", type=Path, default=COMFY_SCRIPTS,
                   help="Path to comfy-service-pipelines/scripts (for run_pipeline + generator)")
    p.add_argument("--allow-non-gemini", action="store_true",
                   help="Permit a non-Gemini oracle model (disables the Gemini-only guards)")
    p.add_argument("--overwrite", action="store_true", help="Regenerate existing images")
    p.add_argument("--skip-generation", action="store_true", help="Validate existing images only")
    p.add_argument("--skip-validation", action="store_true", help="Generate images only")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    load_env(args.env_file)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["KINK_ORACLES_CACHE_DIR"] = args.cache_dir

    # Make run_pipeline + the generator importable.
    sys.path.insert(0, str(args.comfy_scripts))
    try:
        from comfy_checkpoint_generator import ComfyCheckpointGenerator
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"Could not import the generator from {args.comfy_scripts}: {error}")
    from kink_oracles import (
        CloneOracleConfig,
        CloneValidationConfig,
        generate_reports,
        load_kinks,
        run_clone_validation,
    )

    # Fail before any (slow) generation if the oracle model can't be used.
    if not args.skip_validation:
        check_model_available(args.model, args.allow_non_gemini)

    checkpoints_dir = args.checkpoints_dir or (args.exp_dir / "output_all")
    checkpoint = args.checkpoint or latest_checkpoint(checkpoints_dir)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    name = checkpoint.stem
    lora_subdir = args.lora_subdir if args.lora_subdir is not None else args.exp_dir.name

    metrics_root = args.metrics_dir or (args.exp_dir / "metrics")
    out_dir = metrics_root / name
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    logger.info("checkpoint: %s (step %d)", checkpoint.name, step_of(checkpoint))
    logger.info("output dir: %s", out_dir)

    avatars_base = ensure_avatars(args.avatars_base)
    logger.info("avatars: %s", avatars_base)

    kinks = load_kinks(top_n=args.top_n)
    logger.info("kinks: %d", len(kinks))

    generator = None
    if not args.skip_generation:
        logger.info("connecting to ComfyUI %s and installing checkpoint into %s ...",
                    args.base_url, args.lora_title)
        generator = ComfyCheckpointGenerator(
            args.pipeline, args.base_url,
            checkpoint=checkpoint, loras_dir=args.loras_dir,
            lora_subdir=lora_subdir, lora_title=args.lora_title, timeout=args.timeout,
        )

    images_info = []
    total = len(kinks)
    for index, kink in enumerate(kinks):
        safe = re.sub(r"[^0-9A-Za-z]+", "_", kink["kink"]).strip("_")
        out = images_dir / f"{index:03d}_{safe}.png"
        ref = avatars_base / f"{index:03d}.png"

        if generator is not None and (args.overwrite or not out.exists()):
            if not ref.exists():
                logger.warning("[%d/%d] missing avatar %s, skipping %s",
                               index + 1, total, ref.name, kink["kink"])
                continue
            try:
                generator.generate(kink["gandalf_prompt"],
                                   reference_image=str(ref), seed=args.seed).save(out)
                logger.info("[%d/%d] %s", index + 1, total, out.name)
            except Exception as error:  # noqa: BLE001 - keep going on a single failure
                logger.error("[%d/%d] FAILED %s: %s", index + 1, total, kink["kink"], error)
                continue

        if out.exists():
            images_info.append({
                "kink": kink["kink"],
                "category": "gandalf",
                "img_path": str(out),
                "prompt": kink["gandalf_prompt"],
                "description": kink.get("description", ""),
                "row_idx": index,
            })

    logger.info("images ready for oracle: %d", len(images_info))
    if args.skip_validation or not images_info:
        return

    config = CloneValidationConfig(oracle=CloneOracleConfig(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        avatar_refs_base=str(avatars_base),
        n_jobs=args.n_jobs,
        model_names=[args.model],
    ))
    results = run_clone_validation(images_info, config)
    guard_no_fallback(results, args.model, args.allow_non_gemini)
    generate_reports(results, out_dir, name)
    logger.info("reports written to %s", out_dir)


if __name__ == "__main__":
    main()
