# Contributing to ReVolta

ReVolta focuses on the configurations in the
[support scope](.github/SUPPORT.md). Inherited
vLLM functionality outside that scope is not a support commitment.

## Issues

Search existing issues and pull requests first. Use the issue template and
include a minimal reproduction, expected and actual behavior, GPU model,
interconnect, host memory, checkpoint and revision, complete launch command,
ReVolta version, and relevant logs. Remove credentials and private data.
Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Pull requests

Keep each change focused and explain the problem, resulting behavior and
support-scope impact. For an upstream fix, link its source commit and describe
adaptations. Preserve authorship, copyright and license notices.

Run relevant tests and `pre-commit run --files <changed files>` in a project
virtual environment managed with `uv`. Give exact commands and results, plus
hardware and configuration for runtime checks. State what remains untested;
GPU validation may require maintainer hardware.

Every pull request runs these checks, and all must pass before it can merge:
public text (no private addresses, local paths or credentials in the PR, its
commit messages or added lines), sign-off and added files (every commit signed
off; no added file over 1 MiB unless listed in
`.github/large-files-allowed.txt`), pre-commit on the changed files, and mypy
for Python 3.12.

Disclose AI assistance. The submitting human must review every changed line,
understand the change, run relevant checks, and be able to explain the result.
Contributions are subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Developer Certificate of Origin

By contributing, you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
Sign off every commit with `git commit -s`, using your own name and email.
Include an appropriate co-author trailer when AI tools assisted the work.
Contributions use Apache-2.0 unless an existing component's license applies.
