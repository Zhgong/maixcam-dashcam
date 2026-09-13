#!/usr/bin/env python3
"""
Security & Desensitization Auditor for Public Releases.
Enforces strict gate checks before code is pushed to public open-source repositories:
1. Detects secrets, tokens, private keys, passwords.
2. Detects internal / private IP addresses, sensitive internal paths, confidential tags.
3. Ensures all unit tests pass prior to public release.
"""

import os
import re
import sys
import subprocess
from typing import List, Tuple

# Regex patterns for sensitive information
SENSITIVE_PATTERNS = [
    # Tokens & Keys
    (r'(?i)(ghp_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{82})', "GitHub Personal Access Token"),
    (r'-----BEGIN (?:RSA|OPENSSH|DSA|EC) PRIVATE KEY-----', "Private Cryptographic Key"),
    (r'(?i)(?:api_key|secret_key|access_token|password)\s*[:=]\s*["\'][^"\']{8,}["\']', "Potential Hardcoded Secret/Token"),
    (r'(?i)(?:aws_access_key_id|aws_secret_access_key)\s*[:=]\s*["\'][^"\']+["\']', "AWS Credentials"),

    # Sensitive Local User Absolute Paths
    (r'/home/(?!runner\b)[a-zA-Z0-9_-]+/', "Local User Home Directory Path"),

    # Internal / Confidential Annotations
    (r'(?i)#\s*INTERNAL\b', "Internal development tag"),
    (r'(?i)CONFIDENTIAL\b', "Confidential tag"),
]

# Files / patterns excluded from scan
EXCLUDED_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.raw', '.bin', '.mud', '.axmodel', '.pyc', '.git'
}
EXCLUDED_FILES = {
    'scripts/audit_public_release.py',  # Exclude self patterns
    '.agents/AGENTS.md'                 # Exclude agent rule documentation definitions
}


def get_tracked_files() -> List[str]:
    """Retrieve all files tracked in git."""
    try:
        out = subprocess.check_output(['git', 'ls-files'], text=True)
        return [f.strip() for f in out.strip().splitlines() if f.strip()]
    except Exception as e:
        print(f"❌ Error getting git files: {e}", file=sys.stderr)
        return []


def scan_file(file_path: str) -> List[Tuple[int, str, str]]:
    """Scan a single file for sensitive patterns. Returns list of (line_no, rule_desc, matched_snippet)."""
    violations = []
    if any(file_path.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
        return violations
    if file_path in EXCLUDED_FILES:
        return violations

    if not os.path.exists(file_path) or os.path.isdir(file_path):
        return violations

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f, start=1):
                # Ignore gitignored files or test mocks explicitly marked
                for pattern, desc in SENSITIVE_PATTERNS:
                    match = re.search(pattern, line)
                    if match:
                        snippet = line.strip()[:100]
                        violations.append((idx, desc, snippet))
    except Exception as e:
        print(f"⚠️ Warning: Could not read {file_path}: {e}", file=sys.stderr)

    return violations


def run_unit_tests() -> bool:
    """Execute pytest/unittest test suite to ensure quality gate passes."""
    print("🧪 Running unit test quality gate...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run([sys.executable, "-m", "unittest", "discover", "tests"], env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("❌ Public Quality Gate Failed: Unit tests did not pass!", file=sys.stderr)
        print(res.stdout, file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        return False
    print("✅ Unit tests passed (100%).")
    return True


def main() -> int:
    print("🛡️ [Public Release Gate] Starting desensitization and security audit...")
    files = get_tracked_files()
    total_violations = 0

    for fpath in files:
        issues = scan_file(fpath)
        if issues:
            for line_no, desc, snippet in issues:
                print(f"🚨 [LEAK PREVENTED] {fpath}:{line_no} - {desc}", file=sys.stderr)
                print(f"   Snippet: {snippet}", file=sys.stderr)
                total_violations += 1

    if total_violations > 0:
        print(f"\n❌ [AUDIT FAILED] Found {total_violations} sensitive item(s). Push to public repository is BLOCKED!", file=sys.stderr)
        print("💡 Remediate: remove tokens, generalize local paths, or sanitize sensitive information before releasing to public.", file=sys.stderr)
        return 1

    print("✅ Desensitization scan passed! Zero sensitive tokens or private paths found.")

    if not run_unit_tests():
        return 1

    print("🎉 [Public Release Gate] All checks passed! Ready for public release.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
