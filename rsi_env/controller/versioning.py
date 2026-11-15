"""Version and lineage management (controller-side, protected from the Improver).

Accepted systems are named ``H<i>/M<j>``: the harness counter increases only when
harness content changes and the model counter only when the model changes, as in
the paper's Table 1 ("Final version (H/M)").
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".tool_outputs")


def dir_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in Path(d).rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(d)).replace("\\", "/").encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def spec_key(spec: dict) -> str:
    return json.dumps({k: v for k, v in sorted(spec.items()) if k != "id"}, sort_keys=True)


class Registry:
    def __init__(self, private_dir: Path):
        self.dir = Path(private_dir)
        self.path = self.dir / "registry.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "versions": [], "harnesses": {}, "models": {}, "candidates": {}, "active": None}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ init
    def initialize(self, harness_dir: Path, model_spec: dict) -> str:
        h = self.register_harness(harness_dir)
        m = self.register_model(model_spec)
        vid = self.add_version(h, m, parent=None, round_=0, improver="seed", note="initial system")
        self.data["active"] = vid
        self.save()
        return vid

    # ------------------------------------------------------------------ components
    def register_harness(self, src: Path) -> str:
        digest = dir_hash(src)
        for hid, rec in self.data["harnesses"].items():
            if rec["hash"] == digest:
                return hid
        hid = f"H{len(self.data['harnesses'])}"
        dst = self.dir / "harnesses" / hid
        shutil.copytree(src, dst, ignore=IGNORE)
        self.data["harnesses"][hid] = {"dir": str(dst), "hash": digest}
        self.save()
        return hid

    def register_model(self, spec: dict) -> str:
        key = spec_key(spec)
        for mid, rec in self.data["models"].items():
            if spec_key(rec) == key:
                return mid
        mid = f"M{len(self.data['models'])}"
        self.data["models"][mid] = {**spec, "id": mid}
        self.save()
        return mid

    def add_candidate(self, cand_id: str, record: dict) -> None:
        self.data["candidates"][cand_id] = record
        self.save()

    def add_version(self, h: str, m: str, parent: str | None, round_: int, improver: str, note: str = "") -> str:
        vid = f"{h}/{m}"
        self.data["versions"].append({"id": vid, "harness": h, "model": m, "parent": parent,
                                      "round": round_, "improver": improver, "note": note,
                                      "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.save()
        return vid

    # ------------------------------------------------------------------ queries
    @property
    def active(self) -> str:
        return self.data["active"]

    def set_active(self, vid: str) -> None:
        self.data["active"] = vid
        self.save()

    def harness_dir(self, hid: str) -> Path:
        return Path(self.data["harnesses"][hid]["dir"])

    def model_spec(self, mid: str) -> dict:
        return dict(self.data["models"][mid])

    def split(self, vid: str) -> tuple[str, str]:
        h, m = vid.split("/")
        return h, m

    def lineage(self) -> list[dict]:
        return list(self.data["versions"])
