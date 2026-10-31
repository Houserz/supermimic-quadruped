"""Download the public motion sources at pinned revisions into ``data/raw/`` (about 470 MB).

* AI4Animation dog motion capture (Zhang et al., SIGGRAPH 2018; CC BY-NC 4.0)
* Tencent Robotics X Labrador motion capture (Lifelike Agility and Play, NMI 2024; research use)
* Xiaomi CyberDog2 motion lists (cyberdog_locomotion; Apache-2.0)

Each file's URL, SHA-256 and licence is in ``configs/sources.yaml``; files already present with
the recorded hash are not downloaded again. Archives are unpacked member by member and only
``.bvh`` files are written, under their base names, so nothing in an archive picks its own path.

  uv run python scripts/data/fetch_sources.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SOURCES_YAML = ROOT / "configs" / "sources.yaml"

# Pinned revisions (2026-10-29).
TENCENT_REPO, TENCENT_SHA = "Tencent-RoboticsX/lifelike-agility-and-play", "cc9eaa9e79b35f1dd03983f28b669365098e30a5"
CYBERDOG_REPO, CYBERDOG_SHA = "MiRoboticsLab/cyberdog_locomotion", "ad79bfdebba17ae803d0ed37ccc0d1c5e3627af3"
AI4A_URL = "https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip"
AI4A_SHA256 = "0ecd909b673cfca5e20c3f568376dcf27d7da3bd153957c2b034d95af366132a"  # not published upstream; recorded 2026-10-29 (Last-Modified 2018-06-02)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url: str, retries: int = 4) -> bytes:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "supermimic-quadruped-fetch"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network hiccups are retried, the last one is raised
            if k == retries - 1:
                raise
            time.sleep(2.0 * (k + 1))
    raise RuntimeError("unreachable")


def github_tree(repo: str, sha: str) -> list[str]:
    data = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"))
    assert not data.get("truncated"), f"tree of {repo} truncated"
    return [e["path"] for e in data["tree"] if e["type"] == "blob"]


def raw_url(repo: str, sha: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"


class Ledger:
    """configs/sources.yaml: one entry per file, keyed by its path under the project root."""

    def __init__(self) -> None:
        self.data = yaml.safe_load(SOURCES_YAML.read_text()) if SOURCES_YAML.exists() else {}
        self.data.setdefault("sources", {})
        self.data.setdefault("files", {})

    def source(self, key: str, **info) -> None:
        self.data["sources"][key] = info

    def known_hash(self, rel: str) -> str | None:
        return (self.data["files"].get(rel) or {}).get("sha256")

    def file(self, rel: str, source: str, url: str, sha: str) -> None:
        self.data["files"][rel] = {"source": source, "url": url, "sha256": sha}

    def save(self) -> None:
        SOURCES_YAML.parent.mkdir(parents=True, exist_ok=True)
        header = ("# Public motion sources: pinned revision, licence and the SHA-256 of every raw file.\n"
                  "# Written by scripts/data/fetch_sources.py; a file whose hash matches is not downloaded again.\n")
        SOURCES_YAML.write_text(header + yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True, width=120))


def fetch_file(ledger: Ledger, source: str, url: str, dest: Path, expect: str | None = None) -> None:
    rel = str(dest.relative_to(ROOT))
    if dest.exists() and ledger.known_hash(rel) == sha256_file(dest):
        return
    blob = get(url)
    digest = sha256_bytes(blob)
    if expect is not None and digest != expect:
        raise SystemExit(f"{url}: sha256 {digest} != expected {expect}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    ledger.file(rel, source, url, digest)
    print(f"  {rel}  {len(blob) / 1e6:.2f} MB")


def fetch_ai4animation(ledger: Ledger) -> None:
    print("ai4animation_dog: MANN dog motion capture (SIGGRAPH 2018)")
    ledger.source("ai4animation_dog", url=AI4A_URL, license="CC BY-NC 4.0 (motion capture data), research/education only",
                  citation="H. Zhang, S. Starke, T. Komura, J. Saito. Mode-Adaptive Neural Networks for Quadruped Motion Control. SIGGRAPH 2018")
    dest_dir = RAW / "ai4animation_dog"
    zpath = dest_dir / "MotionCapture.zip"
    fetch_file(ledger, "ai4animation_dog", AI4A_URL, zpath, AI4A_SHA256)
    bvh_dir = dest_dir / "bvh"
    bvh_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            base = Path(info.filename).name
            if info.is_dir() or not base.lower().endswith(".bvh") or base.startswith("."):
                continue
            out = bvh_dir / base
            data = z.read(info)
            if not out.exists() or sha256_file(out) != sha256_bytes(data):
                out.write_bytes(data)
            ledger.file(str(out.relative_to(ROOT)), "ai4animation_dog", f"{AI4A_URL}#{info.filename}", sha256_bytes(data))


def fetch_tencent(ledger: Ledger) -> None:
    print("tencent_labrador: Lifelike Agility and Play (Tencent Robotics X, NMI 2024)")
    ledger.source("tencent_labrador", repo=f"https://github.com/{TENCENT_REPO}", commit=TENCENT_SHA, path="data/raw_mocap_data",
                  license="MIT (repository LICENSE.txt); README: code and data for research purpose only")
    paths = [p for p in github_tree(TENCENT_REPO, TENCENT_SHA) if p.startswith("data/raw_mocap_data/") and p.endswith(".bvh")]
    for p in sorted(paths):
        fetch_file(ledger, "tencent_labrador", raw_url(TENCENT_REPO, TENCENT_SHA, p), RAW / "tencent_labrador" / Path(p).name)
    fetch_file(ledger, "tencent_labrador", raw_url(TENCENT_REPO, TENCENT_SHA, "LICENSE.txt"), RAW / "tencent_labrador" / "LICENSE.txt")


def fetch_cyberdog(ledger: Ledger) -> None:
    print("cyberdog2: Xiaomi CyberDog2 motion lists")
    ledger.source("cyberdog2", repo=f"https://github.com/{CYBERDOG_REPO}", commit=CYBERDOG_SHA, license="Apache-2.0",
                  path="control/motion_list/cyberdog2")
    tree = github_tree(CYBERDOG_REPO, CYBERDOG_SHA)
    wanted = [p for p in tree if p.startswith("control/motion_list/cyberdog2/") and p.endswith(".toml")]
    wanted += ["LICENSE", "common/config/cyberdog2-description.yaml"]
    for p in sorted(wanted):
        rel = Path(p)
        if p.startswith("control/motion_list/cyberdog2/"):
            rel = Path("motion_list") / rel.relative_to("control/motion_list/cyberdog2")
        else:
            rel = Path(rel.name)
        fetch_file(ledger, "cyberdog2", raw_url(CYBERDOG_REPO, CYBERDOG_SHA, p), RAW / "cyberdog2" / rel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=["ai4animation_dog", "tencent_labrador", "cyberdog2"])
    args = ap.parse_args()
    ledger = Ledger()
    steps = {"ai4animation_dog": fetch_ai4animation, "tencent_labrador": fetch_tencent, "cyberdog2": fetch_cyberdog}
    for key, fn in steps.items():
        if args.only and key not in args.only:
            continue
        fn(ledger)
        ledger.save()
    print(f"ledger: {SOURCES_YAML.relative_to(ROOT)} ({len(ledger.data['files'])} files)")


if __name__ == "__main__":
    main()
