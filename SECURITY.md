# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private reporting instead: the **Security** tab of this repository ->
**Report a vulnerability**. That creates a private security advisory visible only to you and the
maintainer, and it is the only reporting channel this project offers.

If private reporting is unavailable to you for some reason, open a public issue that says only
"security report, requesting a private channel" - with no details - and a private advisory will be
opened for you.

## What helps

A report is easiest to act on when it contains:

- The release tag, or the SHA-256 of the `moe-direct-v0.2-win-x64.zip` you are running.
- Windows version, and whether the launcher was started from `Start-MoeDirect.cmd` or directly.
- Exact steps to reproduce, and what an attacker gains if it works.
- Any relevant lines from `%LOCALAPPDATA%\MoE-Direct\logs\` - please skim first, since those files
  record local paths that usually contain your Windows user name.

Please do not test against machines that are not yours, and there is no need for destructive
testing to make a point - a description of the mechanism is enough.

## Scope

**In scope** - the parts this project actually builds and ships:

- The PowerShell launcher and its `.cmd` entry point (`Start-MoeDirect.ps1`,
  `Start-MoeDirect.cmd`) - argument and path handling, the state and log files it writes under
  `%LOCALAPPDATA%\MoE-Direct\`, the child process it spawns, and the loopback binding.
- The repacker (`repack_experts.py`) and the integrity gates around it - anything that lets
  unverified expert data be served, or lets a verify report be accepted when it should not be.
- The bundle integrity chain - the internal per-file SHA manifest, the catalog
  (`models.json`) and expectation files, and the checks that consume them.
- The release packaging itself, including anything that would let a modified bundle pass the
  launcher's checks.
- This project's patches to the engine (the direct-read path, the expert cache, the prefetch
  logic).

**Out of scope** - real issues, but not ones this project can fix:

- **Upstream llama.cpp vulnerabilities.** Report those to
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) so every downstream project
  benefits. If an upstream issue is made materially worse by something specific to MoE-Direct,
  that part is in scope here.
- **Third-party components bundled unmodified** - the CPython runtime shipped for the repacker,
  CUDA runtime DLLs, and other vendor binaries. Report to their upstreams; tell us too if a
  version bump is the fix, and it will be picked up in the next release.
- **Model weights and anything derived from model behaviour.** This project never distributes
  weights, and prompt-level behaviour of a model you supplied is not a vulnerability in this
  software.
- **SmartScreen, Smart App Control or antivirus warnings on the unsigned build.** These are
  documented, expected behaviour for a new unsigned file, not a defect - see the README. A
  *false-positive malware detection* is worth telling us about, because the useful response is a
  submission to the vendor with the exact file.
- **Findings that require an attacker who already has administrator rights, or who can already
  write into your bundle folder or your model files.** At that point the machine is already
  theirs. Note the deliberate exception: the launcher's own integrity gates are in scope, because
  refusing to run on a modified bundle is a thing this project claims to do.
- Anything reachable only by hand-editing files that the launcher and the engine produce for
  themselves.

## Supported versions

| Version | Supported |
|---|---|
| Latest release | Yes |
| Anything older | No |

There are no backports and no long-term-support branches. Fixes ship in the next release, and the
remedy for an older build is to update.

## What to expect from us

This is a **public preview maintained by one person in their own time**. Being honest about that up
front is more useful than a service-level promise that would not be kept:

- Reports are read, and acknowledged as soon as the maintainer is able to - not within a fixed
  number of hours.
- Confirmed issues are fixed as fast as is practical for a single maintainer, and the fix ships in
  a release.
- You will be told what was decided, including when a report is judged out of scope or not a
  vulnerability, with the reasoning.
- Credit in the advisory and release notes if you want it; say so in the report, and say how you
  want to be named.
- Coordinated disclosure is appreciated - please hold public details until a fix is available or
  it becomes clear one is not coming. No timeline is demanded of you, and no legal threat will
  ever be made about a good-faith report.

**There is no bug bounty.** No payment, no swag, no rewards program. This project has no funding,
and pretending otherwise would waste your time.
