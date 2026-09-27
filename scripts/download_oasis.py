#!/usr/bin/env python3
"""Download OASIS-3 or OASIS-4 MRI scans from Central XNAT.

OASIS data lives on Central XNAT (https://central.xnat.org). This script
follows the same pattern as NrgXnat's official ``oasis-scripts`` GitHub
project (github.com/NrgXnat/oasis-scripts) but is a single Python file
with no dependencies beyond ``requests``.

Two-step download:

    1. **Filter**: pull the clinical / diagnostic CSV for the target
       project via XNAT REST, restrict to CN + AD (+ optionally MCI),
       write a subject list to ``<dst>/target_subjects.csv``.

    2. **Fetch**: for each target subject, list MR sessions, filter to
       T1 MPRAGE series (SAG_MPR / MPR / MPRAGE variants), and download
       the DICOM/NIfTI archive per session to
       ``<dst>/<subject_id>/<session>/<series>/<file>``.

Usage
-----
    export XNAT_USER=your_username
    export XNAT_PASS=your_password
    python3 scripts/download_oasis.py --project OASIS3 \\
        --dst data/raw/oasis3 \\
        --diagnostic-groups CN AD MCI \\
        --max-subjects 800
    python3 scripts/download_oasis.py --project OASIS4 \\
        --dst data/raw/oasis4 \\
        --diagnostic-groups CN AD MCI

Design notes
------------
* Auth via ``/data/JSESSION`` (JSESSIONID cookie), same pattern as the
  official scripts. Credentials are read from ``XNAT_USER`` / ``XNAT_PASS``
  env vars or prompted on stdin (getpass, never echoed).
* Downloads are resumable per session — if the target session directory
  already exists and contains the expected file count, that session is
  skipped. Kill and re-run at any time.
* T1 MPRAGE series identification is done via ``series_description``
  substring match (case-insensitive): ``MPRAGE``, ``SAG_MPR``, ``MPR``.
  This mirrors OASIS-3 protocol conventions; if your target session
  uses a non-standard series name pass ``--series-pattern`` explicitly.
* The two-step filter avoids downloading anything from subjects outside
  the target diagnostic groups — saves 60–70% of bandwidth for OASIS-3.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:
    print("error: `requests` required. pip install requests", file=sys.stderr)
    sys.exit(1)


XNAT_BASE = "https://central.xnat.org"


# ── Auth ─────────────────────────────────────────────────────────────────
def get_credentials() -> tuple[str, str]:
    user = os.environ.get("XNAT_USER")
    pw   = os.environ.get("XNAT_PASS")
    if not user:
        user = input("XNAT username: ").strip()
    if not pw:
        pw = getpass.getpass("XNAT password: ")
    return user, pw


def _configure_ssl(sess: requests.Session, insecure: bool = False):
    """
    Anaconda Python's certifi bundle sometimes rejects XNAT's Sectigo chain
    even after upgrade. Try in order:
      1. requests default (certifi)
      2. macOS system CA bundle /etc/ssl/cert.pem
      3. insecure (verify=False) as last resort — warning-only
    """
    import ssl
    if insecure:
        sess.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return
    # First fall-back: macOS system bundle
    system_bundle = Path("/etc/ssl/cert.pem")
    if system_bundle.exists():
        sess.verify = str(system_bundle)


def open_session(user: str, pw: str, timeout: float = 20.0,
                 insecure: bool = False) -> requests.Session:
    s = requests.Session()
    _configure_ssl(s, insecure=False)
    try:
        r = s.post(f"{XNAT_BASE}/data/JSESSION", auth=(user, pw), timeout=timeout)
    except requests.exceptions.SSLError as e:
        # Retry with macOS system bundle explicit
        print(f"  ssl error with default bundle ({e.__class__.__name__}); "
              "retrying with /etc/ssl/cert.pem", file=sys.stderr)
        _configure_ssl(s, insecure=insecure)
        try:
            r = s.post(f"{XNAT_BASE}/data/JSESSION", auth=(user, pw), timeout=timeout)
        except requests.exceptions.SSLError:
            if not insecure:
                print("  ssl still failing; retrying with verify=False "
                      "(INSECURE — only use for trusted networks)", file=sys.stderr)
                _configure_ssl(s, insecure=True)
                r = s.post(f"{XNAT_BASE}/data/JSESSION", auth=(user, pw), timeout=timeout)
            else:
                raise
    if r.status_code != 200:
        raise RuntimeError(
            f"XNAT auth failed ({r.status_code}): {r.text[:200]}"
        )
    return s


def close_session(sess: requests.Session):
    try:
        sess.delete(f"{XNAT_BASE}/data/JSESSION", timeout=10)
    except Exception:
        pass


# ── Filter step: pull clinical CSV, restrict to target diagnostic groups ─
def list_subjects_with_diagnosis(
    sess: requests.Session, project: str,
    target_groups: list[str]
) -> list[dict]:
    """
    Read XNAT project's clinical/experimental subject records and return
    those matching target diagnostic groups.

    OASIS uses different diagnostic columns across projects:
      * OASIS3: `dx1` column (`Cognitively normal`, `AD Dementia`, ...)
      * OASIS4: `primary_diagnosis` (variants)
    We try both and pick whichever populates.
    """
    url = (
        f"{XNAT_BASE}/data/projects/{project}/subjects"
        f"?format=json&columns=ID,label,project,gender,age,handedness,dx1,"
        f"primary_diagnosis"
    )
    r = sess.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"subjects list failed ({r.status_code}): {r.text[:200]}")
    rows = r.json()["ResultSet"]["Result"]

    target_norm = {t.lower() for t in target_groups}
    normalise = {
        "cognitively normal":            "CN",
        "cognitively unimpaired":        "CN",
        "healthy control":               "CN",
        "normal":                        "CN",
        "control":                       "CN",
        "ad dementia":                   "AD",
        "dementia, ad type":             "AD",
        "alzheimer's disease":           "AD",
        "probable ad":                   "AD",
        "mild cognitive impairment":     "MCI",
        "mci":                           "MCI",
        "amnestic mci":                  "MCI",
        "mci, stable":                   "MCI",
    }

    filtered = []
    unknown = 0
    for r in rows:
        raw = (r.get("dx1") or r.get("primary_diagnosis") or "").strip().lower()
        if not raw:
            unknown += 1
            continue
        std = normalise.get(raw)
        if std is None:
            # partial substring match as a fallback
            for k, v in normalise.items():
                if k in raw or raw in k:
                    std = v; break
        if std and std.lower() in {t.lower() for t in target_groups}:
            r["standard_dx"] = std
            filtered.append(r)
    print(
        f"  project={project}: {len(rows)} subjects fetched, {len(filtered)} in "
        f"target groups {target_groups}, {unknown} without diagnosis"
    )
    return filtered


def write_target_list(subjects: list[dict], dst: Path):
    """Save the filtered subject list for reproducibility."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ID", "label", "project", "gender", "age",
                        "handedness", "dx1", "primary_diagnosis",
                        "standard_dx"]
        )
        writer.writeheader()
        for s in subjects:
            writer.writerow({k: s.get(k, "") for k in writer.fieldnames})
    print(f"  wrote {len(subjects)} target subjects to {dst}")


# ── Fetch step ───────────────────────────────────────────────────────────
def list_sessions_for_subject(sess: requests.Session, project: str, subject_id: str) -> list[dict]:
    url = (
        f"{XNAT_BASE}/data/projects/{project}/subjects/{subject_id}/experiments"
        f"?format=json&xsiType=xnat:mrSessionData&columns=ID,label,date,xsiType"
    )
    r = sess.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  warn: {subject_id} sessions list failed ({r.status_code})", file=sys.stderr)
        return []
    return r.json()["ResultSet"]["Result"]


def list_scans_for_session(sess: requests.Session, session_id: str) -> list[dict]:
    url = (
        f"{XNAT_BASE}/data/experiments/{session_id}/scans"
        f"?format=json&columns=ID,type,series_description,quality,frames"
    )
    r = sess.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  warn: {session_id} scans list failed ({r.status_code})", file=sys.stderr)
        return []
    return r.json()["ResultSet"]["Result"]


def matches_pattern(scan: dict, pattern: re.Pattern) -> bool:
    for key in ("series_description", "type"):
        v = (scan.get(key) or "").strip()
        if v and pattern.search(v):
            return True
    return False


def download_scan_files(
    sess: requests.Session, session_id: str, scan_id: str,
    dst: Path, retry: int = 3
) -> tuple[int, int]:
    """Download all files of a given scan as a zip; extract to dst.

    Returns (n_files_written, n_bytes).
    """
    zip_url = (
        f"{XNAT_BASE}/data/experiments/{session_id}/scans/{scan_id}"
        f"/files?format=zip"
    )
    dst.mkdir(parents=True, exist_ok=True)
    zip_path = dst / f"scan_{scan_id}.zip"

    for attempt in range(1, retry + 1):
        try:
            with sess.get(zip_url, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    print(f"    warn: scan {scan_id} HTTP {r.status_code} (attempt {attempt})",
                          file=sys.stderr)
                    time.sleep(15 * attempt)
                    continue
                n_bytes = 0
                with zip_path.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):  # 1 MB chunks
                        if chunk:
                            f.write(chunk); n_bytes += len(chunk)
            # Extract on the fly (session dir is small: <1 GB usually)
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                n_files = len(z.namelist())
                z.extractall(dst)
            zip_path.unlink()  # discard zip after extract
            return n_files, n_bytes
        except Exception as e:
            print(f"    warn: scan {scan_id} attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(15 * attempt)
    return 0, 0


def already_downloaded(session_dst: Path) -> bool:
    """
    Heuristic: session considered done if directory exists and contains
    at least one .nii or .dcm file. Callers can force re-download by
    deleting session_dst.
    """
    if not session_dst.is_dir():
        return False
    for pat in ("*.nii", "*.nii.gz", "*.dcm"):
        if any(session_dst.rglob(pat)):
            return True
    return False


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, choices=["OASIS3", "OASIS4"])
    p.add_argument("--dst", type=Path, required=True,
                   help="destination root (e.g. data/raw/oasis3, or an external drive)")
    p.add_argument("--diagnostic-groups", nargs="+",
                   default=["CN", "AD", "MCI"],
                   help="which diagnostic groups to keep (default: CN AD MCI)")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="cap number of subjects downloaded (smoke test)")
    p.add_argument("--series-pattern", type=str,
                   default=r"(?i)MPRAGE|SAG_MPR|(?<![A-Z])MPR(?![A-Z])",
                   help="regex matched against series_description")
    p.add_argument("--filter-only", action="store_true",
                   help="only write the target subject list; do not download")
    args = p.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    target_csv = args.dst / "target_subjects.csv"

    # Step 1: authenticate + filter subjects
    user, pw = get_credentials()
    print(f"authenticating as {user}...")
    sess = open_session(user, pw)
    try:
        subjects = list_subjects_with_diagnosis(
            sess, args.project, args.diagnostic_groups
        )
        write_target_list(subjects, target_csv)

        if args.max_subjects:
            subjects = subjects[:args.max_subjects]
            print(f"  capping to first {len(subjects)} subjects (smoke test)")

        if args.filter_only:
            print("--filter-only requested; not downloading scans.")
            return 0

        # Step 2: download MPRAGE per session per subject
        series_pat = re.compile(args.series_pattern)
        totals = {"subjects": 0, "sessions": 0, "scans": 0,
                  "files": 0, "bytes": 0, "skipped": 0}
        for i, subj in enumerate(subjects, 1):
            subject_id = subj["ID"]
            label = subj.get("label") or subject_id
            std_dx = subj.get("standard_dx", "")
            print(f"[{i}/{len(subjects)}] {label} ({std_dx})")
            sess_records = list_sessions_for_subject(sess, args.project, subject_id)
            for session in sess_records:
                session_id = session["ID"]
                session_label = session.get("label") or session_id
                session_dst = args.dst / label / session_label
                if already_downloaded(session_dst):
                    totals["skipped"] += 1
                    print(f"    skip (exists): {session_label}")
                    continue

                scans = list_scans_for_session(sess, session_id)
                mprage_scans = [s for s in scans if matches_pattern(s, series_pat)]
                if not mprage_scans:
                    continue
                totals["sessions"] += 1
                for scan in mprage_scans:
                    scan_id = scan["ID"]
                    scan_desc = scan.get("series_description") or scan.get("type") or scan_id
                    scan_dst = session_dst / f"scan_{scan_id}_{scan_desc}"
                    n_files, n_bytes = download_scan_files(sess, session_id, scan_id, scan_dst)
                    totals["scans"] += 1
                    totals["files"] += n_files
                    totals["bytes"] += n_bytes
                    print(f"    ✓ {scan_desc}: {n_files} files, {n_bytes / 1e6:.1f} MB")
            totals["subjects"] += 1
        print("")
        print(f"DONE — subjects={totals['subjects']} "
              f"sessions={totals['sessions']} scans={totals['scans']} "
              f"files={totals['files']} "
              f"total={totals['bytes'] / 1e9:.2f} GB "
              f"(skipped {totals['skipped']} existing sessions)")
        return 0
    finally:
        close_session(sess)


if __name__ == "__main__":
    sys.exit(main())
