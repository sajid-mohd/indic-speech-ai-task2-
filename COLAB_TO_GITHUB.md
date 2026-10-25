# Publish this project from Google Colab to GitHub

## Included in this release

- Source code under `src/` and `scripts/`
- Cleared Colab notebook `Indic_Speech_AI_Research_Submission.ipynb`
- Final report `FINAL_RESEARCH_REPORT.md`
- README, requirements, unit tests, and aggregate CSV/JSON results
- A push guide in `GITHUB_SUBMISSION.md`

Private row-level QA files, human review workbooks, audio, model weights/caches, and credentials are intentionally not included.

## Steps

1. Create an empty repository on GitHub (do not initialize it with a README).
2. In Colab, upload this ZIP. Extract it so `/content/speech_ai_assignment` exists.
3. Open `/content/speech_ai_assignment/GITHUB_SUBMISSION.md` and run the provided push cell. When prompted, enter the empty repository HTTPS URL, your GitHub username, and a fine-grained token with **Contents: Read and write** permission for that repository. The token is requested through a hidden prompt and should never be placed in a notebook cell or committed.
4. After pushing, open the repository on GitHub and verify the README, notebook, report, code, and aggregate results.

Alternatively, for only the notebook, use Colab **File → Save a copy in GitHub**. That publishes the notebook, not the complete source-code project.

## Important

The response-quality rates are AI-assisted estimates, not independent human annotations. The report says this explicitly. Do not fill the official `human_*` columns with AI-generated labels and call them human-reviewed.
