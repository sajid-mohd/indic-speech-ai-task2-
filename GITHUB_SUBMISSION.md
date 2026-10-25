# Publish from Google Colab to GitHub

## What to publish

Publish the source code, cleaned notebook, README, final report, unit tests, dependency files, and **aggregate** result summaries. Do not publish Hugging Face/GitHub tokens, model weights/caches, raw audio, or row-level QA files containing examples from the gated Hindi speech-instruct dataset unless the dataset terms explicitly allow redistribution. The project's `.gitignore` excludes those row-level files; aggregate summaries remain tracked.

## One-time setup

1. In GitHub, create an **empty** repository, e.g. `indic-speech-ai-task2` (do not initialize with README/license/gitignore if you want the simplest first push).
2. In Colab, upload/extract `speech_ai_research_submission.zip`, or use the `speech_ai_assignment` project directory that contains this repository's files.
3. Run the push cell below, entering the repository HTTPS URL and a GitHub token privately at the prompts. A fine-grained token should have **Contents: Read and write** permission on this repository. Never paste the token into notebook source or commit it.

## Push cell for Colab

```python
from pathlib import Path
from getpass import getpass
import re, os, subprocess, stat

# Change this if you named the repository differently.
repo_url = input("Empty GitHub repository HTTPS URL: ").strip()
if not repo_url.startswith("https://github.com/") or not repo_url.endswith(".git"):
    raise ValueError("Use an HTTPS URL such as https://github.com/USERNAME/indic-speech-ai-task2.git")

project = Path("/content/speech_ai_assignment")
required = [
    "README.md", "FINAL_RESEARCH_REPORT.md", "SUBMISSION_STATUS.md",
    "Indic_Speech_AI_Research_Submission.ipynb", "scripts/evaluate_fleurs.py",
    "scripts/evaluate_speech_qa.py", "src/speech_ai/pipeline.py",
    "results/asr_validation_summary.csv", "results/asr_test_summary.csv",
    "results/adaptive_threshold_sweep.csv", "results/adaptive_threshold_selected_by_condition.csv",
    "results/qa_run_summary.json", "results/qa_ai_assisted_review_summary.json",
    "results/reproducibility_check.json", "results/submission_metrics.json",
]
missing = [p for p in required if not (project / p).is_file()]
if missing:
    raise FileNotFoundError("Project package is incomplete:\n" + "\n".join(missing))

# Refuse to publish obvious secrets and row-level gated QA data.
token_pattern = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
for p in project.rglob("*"):
    if (not p.is_file() or ".git" in p.parts or p.name == "GITHUB_SUBMISSION.md"
            or p.suffix.lower() not in {".py", ".md", ".ipynb", ".json", ".txt"}):
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if token_pattern.search(text):
        raise RuntimeError(f"Potential credential token found in {p}; inspect before publishing.")

for banned in [
    "results/qa_indic_transcribe_cpu_100_20261024/qa_predictions_for_review.csv",
    "results/qa_indic_transcribe_cpu_100_20261024/qa_predictions_ai_assessed.csv",
]:
    if (project / banned).is_file():
        print("Note: .gitignore should exclude private row-level file:", banned)

username = input("GitHub username: ").strip()
token = getpass("GitHub fine-grained token (hidden): ").strip()
if not username or not token:
    raise ValueError("GitHub username and token are required to push.")

# Never place token in the remote URL or write it into the repository.
askpass = Path("/tmp/github_askpass_submission.sh")
askpass.write_text(
    '#!/bin/sh\ncase "$1" in *Username*) printf "%s" "$GITHUB_USERNAME" ;; *) printf "%s" "$GITHUB_TOKEN" ;; esac\n',
    encoding="utf-8"
)
askpass.chmod(0o700)
env = os.environ.copy()
env["GIT_ASKPASS"] = str(askpass)
env["GIT_TERMINAL_PROMPT"] = "0"
env["GITHUB_USERNAME"] = username
env["GITHUB_TOKEN"] = token

try:
    def git(*args, check=True):
        print("$ git", *args)
        return subprocess.run(["git", *args], cwd=project, env=env, check=check, text=True)

    if not (project / ".git").exists():
        git("init", "-b", "main")
    git("config", "user.name", username)
    # GitHub will use the authenticated account; supply a noreply address if you prefer.
    git("config", "user.email", f"{username}@users.noreply.github.com")

    remotes = subprocess.run(["git", "remote"], cwd=project, text=True, capture_output=True, check=True).stdout.split()
    if "origin" in remotes:
        git("remote", "set-url", "origin", repo_url)
    else:
        git("remote", "add", "origin", repo_url)

    # .gitignore prevents raw row-level QA predictions from being staged.
    git("add", "-A")
    status = subprocess.run(["git", "status", "--short"], cwd=project, text=True, capture_output=True, check=True)
    print(status.stdout or "No staged changes.")
    if status.stdout.strip():
        git("commit", "-m", "Submit reproducible Indic speech AI evaluation")

    git("push", "-u", "origin", "main")
    print("Push completed. Open:", repo_url.removesuffix(".git"))
finally:
    env.pop("GITHUB_TOKEN", None)
    env.pop("GITHUB_USERNAME", None)
    del token
    try: askpass.unlink()
    except FileNotFoundError: pass
```

The push cell expects the project directory to be `/content/speech_ai_assignment`, which is what the included notebook's extraction cell creates. It uses an HTTPS token only for the current git subprocess, not as a URL or committed file.
