#!/usr/bin/env python3
"""Submission validator that runs in GitHub Actions and local organizer environments.

Ensures participant models are trained, sealed, documented, and conform strictly
to tournament rules and PyTorch tensor architectures.

Tamper-proof: Detects modifications to the CI/CD workflow and enforces official rules.
"""
import ast
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
STARTER_CHECKPOINT_SHA = "579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7"
STARTER_AGENT_ID = "shryssssss/v13"
EXPECTED_PARAMS = 26181
FORBIDDEN_MODULES = {"socket", "subprocess", "requests", "urllib", "http", "ftplib", "multiprocessing"}
UPSTREAM_WORKFLOW_URL = "https://raw.githubusercontent.com/Model-Wars/Model-Wars-Base-Repository/main/.github/workflows/validate.yml"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ci_tampering(root: Path):
    """Detects if participant altered .github/workflows/validate.yml."""
    local_wf = root / ".github" / "workflows" / "validate.yml"
    if not local_wf.exists():
        sys.exit("FAIL: Disqualified! .github/workflows/validate.yml has been deleted.")

    try:
        req = urllib.request.Request(
            UPSTREAM_WORKFLOW_URL,
            headers={"User-Agent": "Model-Wars-Validator/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            upstream_bytes = resp.read()

        local_sha = sha256_file(local_wf)
        upstream_sha = hashlib.sha256(upstream_bytes).hexdigest()

        if local_sha != upstream_sha:
            sys.exit(
                "FAIL: DISQUALIFIED!\n"
                "Tampering detected: .github/workflows/validate.yml does not match the official tournament workflow.\n"
                "Participants are not permitted to modify CI/CD validation scripts."
            )
        print("[✓] CI/CD pipeline integrity verified against official tournament upstream.")
    except urllib.error.URLError as e:
        print(f"[!] Warning: Network check skipped ({e}); proceeding with local validation.")


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and sys.argv[1] != "." else (
        Path(".").resolve() if Path("bot.toml").exists() else DEFAULT_ROOT
    )
    print(f"[*] Validating repository at: {root}")

    # Check if running on the base starter template repository itself
    gh_repo = os.environ.get("GITHUB_REPOSITORY", "").lower()
    is_base_template = gh_repo.endswith("model-wars-base-repository") or gh_repo.endswith("rlbot-v13-starter")

    if is_base_template and not (root / "submission.json").exists():
        print("[*] Base starter template detected in CI. Verifying baseline integrity...")
        sys.path.insert(0, str(root / "tools"))
        import kit
        kit.verify(root)
        print("[✓] Baseline template integrity confirmed. Ready for participant use.")
        sys.exit(0)

    # 1. Tamper Detection Check (Participant cannot modify validate.yml)
    verify_ci_tampering(root)

    # 2. Submission registration check
    sub_file = root / "submission.json"
    if not sub_file.exists():
        sys.exit(
            "FAIL: submission.json missing!\n"
            "Participants must initialize their submission using:\n"
            "  python tools/kit.py init --name \"<Bot Name>\" --agent-id \"<author/bot>\"\n"
            "and then seal their weights using tools/kit.py seal."
        )

    submission = json.loads(sub_file.read_text(encoding="utf-8"))

    # 3. Identity Check
    bot_toml = (root / "bot.toml").read_text(encoding="utf-8")
    if 'name = "V13"' in bot_toml:
        sys.exit("FAIL: Bot name in bot.toml is still 'V13'. Participants must use a custom bot name.")

    match = re.search(r'agent_id\s*=\s*"([^"]+)"', bot_toml)
    if not match:
        sys.exit("FAIL: Could not find agent_id in bot.toml.")
    agent_id = match.group(1).strip()
    if agent_id == STARTER_AGENT_ID or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", agent_id):
        sys.exit(f"FAIL: agent_id in bot.toml ('{agent_id}') is invalid or still set to starter ID '{STARTER_AGENT_ID}'.")

    if submission.get("agent_id") != agent_id:
        sys.exit(f"FAIL: agent_id mismatch between bot.toml ('{agent_id}') and submission.json ('{submission.get('agent_id')}').")

    # 4. Model Weight & Anti-Plagiarism Check
    model_path = root / "model" / "V13.pt"
    if not model_path.exists():
        sys.exit("FAIL: Model weights file model/V13.pt is missing.")

    # Git LFS pointer check
    if model_path.stat().st_size < 10240:
        content_sample = model_path.read_bytes()[:100]
        if b"git-lfs" in content_sample:
            sys.exit(
                "FAIL: model/V13.pt is an unresolved Git LFS text pointer (<1KB)!\n"
                "Ensure Git LFS is installed and pulled: run 'git lfs pull'."
            )
        sys.exit(f"FAIL: model/V13.pt is suspiciously small ({model_path.stat().st_size} bytes < 10KB).")

    model_sha = sha256_file(model_path)
    if model_sha == STARTER_CHECKPOINT_SHA:
        sys.exit(
            "FAIL: Checkpoint SHA matches unmodified starter weights!\n"
            "Submissions must contain participant-trained model weights. No training was detected."
        )

    # 5. PIN & Manifest Integrity
    runtime_code = (root / "src" / "runtime.py").read_text(encoding="utf-8")
    pin_match = re.search(r"PIN\s*=\s*'([0-9a-f]{64})'", runtime_code)
    if not pin_match or pin_match.group(1) != model_sha:
        sys.exit(f"FAIL: PIN in src/runtime.py does not match actual model/V13.pt SHA256 ({model_sha}). Run tools/kit.py seal.")

    # 6. Provenance & Results Documentation
    provenance = submission.get("training_provenance")
    if not provenance or not isinstance(provenance, dict):
        sys.exit(
            "FAIL: training_provenance in submission.json is missing or empty.\n"
            "Run tools/kit.py seal with --provenance containing honest experiment details."
        )

    results_md = root / "docs" / "PARTICIPANT_RESULTS.md"
    if not results_md.exists() or len(results_md.read_text(encoding="utf-8").strip()) < 50:
        sys.exit(
            "FAIL: docs/PARTICIPANT_RESULTS.md is missing or too brief (< 50 chars).\n"
            "Participants must document their training procedure, validation metrics, and GUI test results."
        )

    # 7. Static Code Security Scan
    for py_file in (root / "src").glob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=py_file.name)
        except SyntaxError as e:
            sys.exit(f"FAIL: Syntax error in {py_file.name}: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in FORBIDDEN_MODULES:
                        sys.exit(f"FAIL: Disallowed import '{a.name}' in {py_file.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN_MODULES:
                    sys.exit(f"FAIL: Disallowed import from '{node.module}' in {py_file.name}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec"}:
                    sys.exit(f"FAIL: Disallowed function call '{node.func.id}' in {py_file.name}")

    # 8. Model Architecture and Tensor Check
    import torch
    from torch import nn

    try:
        loaded = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        sys.exit(f"FAIL: Failed to load PyTorch model with weights_only=True: {e}")

    if not isinstance(loaded, dict) or "state_dict" not in loaded:
        sys.exit("FAIL: Checkpoint dictionary must contain 'state_dict'.")

    state_dict = loaded["state_dict"]
    for k, v in state_dict.items():
        if not torch.isfinite(v).all():
            sys.exit(f"FAIL: Tensor '{k}' contains non-finite (NaN/Inf) values.")

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(13, 64), nn.ReLU())
            self.gru = nn.GRU(64, 64, num_layers=1, batch_first=True, bidirectional=False)
            self.head = nn.Linear(64, 5)

        def forward(self, x, h=None):
            z, h = self.gru(self.encoder(x).unsqueeze(0), h)
            out = self.head(z.squeeze(0))
            return out[:, :4], torch.tanh(out[:, 4]), h

    model = Policy().eval()
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        sys.exit(f"FAIL: Model weights do not match required 13D GRU architecture: {e}")

    num_params = sum(p.numel() for p in model.parameters())
    if num_params != EXPECTED_PARAMS:
        sys.exit(f"FAIL: Model parameter count ({num_params}) does not match expected ({EXPECTED_PARAMS}).")

    # Cosine Similarity Check against baseline starter weights (Anti-Noise Cheat)
    try:
        import io
        import urllib.request
        starter_req = urllib.request.Request(
            "https://raw.githubusercontent.com/Model-Wars/Model-Wars-Base-Repository/main/model/V13.pt",
            headers={"User-Agent": "Model-Wars-Validator/1.0"}
        )
        with urllib.request.urlopen(starter_req, timeout=10) as resp:
            starter_loaded = torch.load(io.BytesIO(resp.read()), map_location="cpu", weights_only=True)
            starter_model = Policy().eval()
            starter_model.load_state_dict(starter_loaded["state_dict"], strict=True)

            v_sub = torch.cat([p.flatten() for p in model.parameters()])
            v_start = torch.cat([p.flatten() for p in starter_model.parameters()])

            cos_sim = torch.nn.functional.cosine_similarity(v_sub, v_start, dim=0).item()
            print(f"    - Baseline Cosine Similarity: {cos_sim:.5f}")
            if cos_sim > 0.999:
                sys.exit(
                    f"FAIL: DISQUALIFIED!\n"
                    f"Model weights have {cos_sim:.5f} cosine similarity (> 0.999) with the starter baseline.\n"
                    f"Trivial noise perturbations of the baseline weights are not permitted. You must train your own policy."
                )
    except Exception as e:
        print(f"[!] Note: Baseline cosine comparison skipped ({e}).")

    with torch.inference_mode():
        dummy_in = torch.randn(1, 13)
        logits, steer, h = model(dummy_in)
        if logits.shape != (1, 4) or steer.shape != (1,) or h.shape != (1, 1, 64):
            sys.exit(f"FAIL: Unexpected inference output shapes: logits={logits.shape}, steer={steer.shape}, h={h.shape}")
        if not (torch.isfinite(logits).all() and torch.isfinite(steer).all()):
            sys.exit("FAIL: Model produced NaN or Inf during test forward pass.")

    print("[✓] SUCCESS: All submission requirements and tensor checks PASSED!")
    print(f"    - Bot Agent ID:  {agent_id}")
    print(f"    - Model SHA-256: {model_sha}")
    print(f"    - Parameters:    {num_params}")


if __name__ == "__main__":
    main()
