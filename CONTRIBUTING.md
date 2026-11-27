# Contributing to ReVolta

This page explains how to report a problem and how to get a change merged. ReVolta supports the configurations in the [support matrix](docs/known-limits.md#support-matrix). Inherited vLLM functionality outside that scope is not a support commitment.

## Report a problem

Search existing issues and pull requests first. Use the bug report form for a problem and the report template for test results or field experience. Include a minimal reproduction, expected and actual behavior, GPU model and count, interconnect, host memory, checkpoint and revision, the complete launch command, the ReVolta version and the relevant logs. Remove credentials and private data.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Development model

- All development happens in this public repository.
- `main` is the trunk. Every change lands through a pull request and is merged with a merge commit; squash and rebase merges are disabled.
- For a release, a `release/0.<minor>.x` branch (for example `release/0.2.x`) is cut from `main` and frozen. The accepted commit is tagged `v0.<minor>.<patch>`, for example `v0.2.0`.
- `prod` is the code the maintainer deploys. It changes only through a pull request from the release branch, and each deployment is tagged `deploy-YYYYMMDD-v<version>`.

Open pull requests against `main` unless a maintainer asks for a fix on a release branch.

## Pull requests

Keep each change focused. Explain the problem, the resulting behavior and any effect on the support matrix. For an upstream fix, link its source (`vllm-project/vllm#NNNN`, `1CatAI/1Cat-vLLM#NNNN` or the commit) and describe your adaptations. Preserve authorship, copyright and license notices.

Run the relevant tests and `pre-commit run --files <changed files>` in a Python 3.12 virtual environment created with `uv`. In the pull request, give the exact commands and results, and the hardware and configuration for runtime checks. State what remains untested; GPU validation may need the maintainer's hardware.

### Required checks

Every pull request into `main`, `prod` or a `release/` branch runs four checks, and all must pass before it can merge:

| Check | What it verifies |
| --- | --- |
| `public text` | The PR title and body, commit messages, author and committer emails, and added lines contain no private IPv4 addresses, `.lan` host names, home-directory or `/mnt` paths, project-internal names, or credentials (GitHub, Hugging Face and PyPI tokens, API keys, AWS keys, private keys). |
| `sign-off and added files` | Every non-merge commit has a `Signed-off-by:` line with its author's email. No added file is larger than 1 MiB unless its path is listed in `.github/large-files-allowed.txt`. |
| `pre-commit (changed files)` | The repository's pre-commit hooks pass on the files the pull request changes. |
| `mypy (3.12)` | mypy for Python 3.12 passes on all files, with the same hook and configuration as upstream vLLM. |

Changes to the dashboard also run a `dashboard CPU` workflow. To run the first two checks locally before you push, install `regex`, fetch this repository's `main` (below, the remote `upstream` points to `Leonccaa/ReVolta`) and run:

```bash
git fetch upstream main
BASE=upstream/main HEAD=HEAD python tools/ci/check_public_text.py
BASE=upstream/main HEAD=HEAD python tools/ci/check_commits.py
```

A local public-text run checks only the generic patterns; the project-internal names come from a repository secret.

## AI-assisted contributions

Disclose AI assistance. The submitting human must review every changed line, understand the change, run the relevant checks and be able to explain the result. See [AGENTS.md](AGENTS.md).

Contributions are subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Developer Certificate of Origin

By contributing, you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/). Sign off every commit with `git commit -s`, using your own name and email. `pre-commit install` also installs a commit-msg hook that adds the sign-off. Add a `Co-authored-by:` trailer when AI tools assisted the work.

Contributions are licensed under Apache-2.0 unless an existing component's license applies.
