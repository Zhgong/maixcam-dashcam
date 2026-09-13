# MaixCAM Dashcam Repository Rules

## Remote & Push Discipline
- **Dual Remotes Definition**:
  - `origin`: Internal private repository (`git@github.com:Zhgong/maixcam-dashcam-internal.git`). Day-to-day development, feature branches, and issues MUST live here.
  - `public`: Public open-source release repository (`git@github.com:Zhgong/maixcam-dashcam.git`).
- **Push Policy**:
  - `git push origin master` is allowed for regular commits (with user approval).
  - **NEVER** push to `public` (`git push public ...`) during day-to-day development. Pushing to `public` is strictly reserved for major milestone releases and requires explicit, separate user instructions.
- **Issues & Planning**:
  - All feature tasks, technical debt, and architecture refactoring issues MUST be created in `origin` (`maixcam-dashcam-internal`), NOT in `public`.

## Security & Desensitization Gate
- **Pre-Push Hook**: Configured at `.git/hooks/pre-push`. Any attempt to `git push public ...` automatically runs `scripts/audit_public_release.py`.
- **Public Audit Checks**:
  1. Secret Scanning: Tokens (`ghp_...`), private keys, passwords.
  2. Privacy Scanning: Private home paths (`/home/...`), confidential tags (`CONFIDENTIAL`, `# INTERNAL`).
  3. Quality Gate: Runs full unit test suite (must be 100% passing).
  4. If any rule is violated, Git immediately aborts the push.
- **CI Cloud Gate**: `.github/workflows/public-security-audit.yml` runs as a secondary fallback on GitHub Actions.
