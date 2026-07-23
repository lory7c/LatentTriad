#!/usr/bin/env python3
"""Download safe skills from MaliciousAgentSkillsBench dataset.

9,512 unique safe repos → extract SKILL.md → organized output.
Parallel download with 20 workers, skip existing, resume-safe.
"""

import csv, os, sys, time, zipfile, io, hashlib, shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

CSV_PATH = "/work/yz/skillprobe/external/baselines/routeguard_comparators/repos/MaliciousAgentSkillsBench/data/skills_dataset.csv"
OUTPUT_DIR = "/work/yz/skillprobe/external/baselines/routeguard_comparators/downloaded_skills"
WORKERS = 20
TIMEOUT = 60
MAX_REPOS = 0  # 0 = all

def find_skill_md(zip_data, skill_name):
    """Extract SKILL.md content from zip bytes. Search for files ending with skill_name/SKILL.md or SKILL.md."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for name in zf.namelist():
                if name.endswith("SKILL.md") or name.endswith("skill.md"):
                    # Match if skill_name appears in the path or if we take any SKILL.md
                    return zf.read(name).decode("utf-8", errors="replace")
    except Exception as e:
        pass
    return None

def download_repo(url, output_dir):
    """Download a repo zip and extract all SKILL.md files."""
    repo_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    repo_dir = Path(output_dir) / repo_hash
    done_file = repo_dir / "_done"

    if done_file.exists():
        return None  # Already downloaded

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SkillProbe/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            zip_data = resp.read()
    except Exception as e:
        print(f"  FAIL {url[:60]}: {e}")
        return None

    try:
        repo_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for name in zf.namelist():
                if name.lower().endswith("skill.md") or name.lower().endswith(".md"):
                    # Extract skill name from path
                    parts = Path(name).parts
                    skill_name = parts[0] if len(parts) > 0 else "unknown"
                    content = zf.read(name).decode("utf-8", errors="replace")
                    # Save under skill name
                    skill_file = repo_dir / f"{skill_name}_{hashlib.md5(name.encode()).hexdigest()[:8]}.md"
                    skill_file.write_text(content, encoding="utf-8")
        done_file.touch()
        return len(list(repo_dir.glob("*.md")))
    except Exception as e:
        print(f"  EXTRACT FAIL {url[:60]}: {e}")
        return None

def main():
    # Load CSV
    print(f"Loading {CSV_PATH}...")
    with open(CSV_PATH) as f:
        rows = list(csv.reader(f))
    header = rows[0]; rows = rows[1:]
    print(f"Total rows: {len(rows)}")

    # Filter safe, deduplicate URLs
    safe = {}
    for row in rows:
        if row[3] == "safe" and row[4] and not row[4].startswith("[REDACTED"):
            url = row[4]
            if url not in safe:
                safe[url] = []
            safe[url].append(row[2])  # skill_name

    urls = sorted(safe.keys(), key=lambda u: -len(safe[u]))
    if MAX_REPOS:
        urls = urls[:MAX_REPOS]

    print(f"Safe repos: {len(urls)}")
    print(f"Total skills: {sum(len(safe[u]) for u in urls)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {WORKERS}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download
    t0 = time.time()
    completed = 0; total = len(urls)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_repo, url, OUTPUT_DIR): url for url in urls}
        for future in as_completed(futures):
            completed += 1
            n_files = future.result()
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / (elapsed / 60)
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] {elapsed/60:.0f}m, {rate:.0f}/min, ETA={eta:.0f}m")

    # Count
    md_files = list(Path(OUTPUT_DIR).rglob("*.md"))
    print(f"\nDone! Downloaded {len(md_files)} MD files in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
