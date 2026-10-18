# Contributing to ReVolta

ReVolta focuses on the configurations in the
[0.1.0 support scope](.github/SUPPORT.md), derived from roadmap W10. Inherited
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

Disclose AI assistance. The submitting human must review every changed line,
understand the change, run relevant checks, and be able to explain the result.
Contributions are subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Developer Certificate of Origin

By contributing, you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
Sign off every commit with `git commit -s`, using your own name and email.
Include an appropriate co-author trailer when AI tools assisted the work.
Contributions use Apache-2.0 unless an existing component's license applies.
