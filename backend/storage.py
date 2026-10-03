"""
storage.py — JSON-backed persistence with advisory file locking.

Layout (under ``data/``)::

    data/
      audio/                      raw audio files (stored separately)
      projects/<project_id>.json  project configuration
      analysis/<file>__<kind>.json  analysis results
      versions/<id>__<n>.json     version snapshots (project history)
      library.json                the audio-file registry / metadata index
      *.lock                      advisory lock files

Concurrency model
-----------------
Every read-modify-write of a JSON document happens under an exclusive advisory
lock (``fcntl.flock``) taken on a sibling ``.lock`` file, and every write is
*atomic* (write to a temp file, ``fsync``, then ``os.replace``).  This means a
crash mid-write can never leave a half-written JSON file, and concurrent
requests (Flask runs threaded by default) serialise correctly.

The "audio files + JSON metadata stay in sync" problem is handled centrally:
deleting a file also removes its analyses and version snapshots under the same
lock, and every mutation goes through :class:`Storage` so the registry and the
on-disk artefacts cannot drift apart.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX fallback
    _HAVE_FCNTL = False


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def new_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------- #
# File lock
# --------------------------------------------------------------------------- #

class FileLock:
    """Advisory exclusive lock implemented with ``fcntl.flock``.

    The lock is re-acquired on a fresh file descriptor each time so that it also
    serialises threads within a single process (flock locks are per open-file-
    description, so two independent ``open`` calls do conflict).
    """

    def __init__(self, path: str, timeout: float = 10.0, poll: float = 0.02):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if _HAVE_FCNTL:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError(f"could not acquire lock on {self.path}")
                time.sleep(self.poll)
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            try:
                if _HAVE_FCNTL:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@contextmanager
def locked(path: str, timeout: float = 10.0):
    """Context-manager shorthand for :class:`FileLock`."""
    lk = FileLock(path, timeout)
    lk.acquire()
    try:
        yield
    finally:
        lk.release()


# --------------------------------------------------------------------------- #
# Atomic JSON document
# --------------------------------------------------------------------------- #

def atomic_write(path: str, data: Any) -> None:
    """Write JSON atomically: temp file + fsync + rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# --------------------------------------------------------------------------- #
# Storage manager
# --------------------------------------------------------------------------- #

class Storage:
    """Central manager for all JSON metadata and audio-file bookkeeping."""

    def __init__(self, root: str):
        self.root = root
        self.audio_dir = os.path.join(root, "audio")
        self.projects_dir = os.path.join(root, "projects")
        self.analysis_dir = os.path.join(root, "analysis")
        self.versions_dir = os.path.join(root, "versions")
        self.library_path = os.path.join(root, "library.json")
        for d in (self.audio_dir, self.projects_dir, self.analysis_dir, self.versions_dir):
            os.makedirs(d, exist_ok=True)

    # -- low-level locked document helpers -------------------------------- #

    def _read_library(self) -> Dict[str, Any]:
        return read_json(self.library_path, {"version": 1, "files": {}})

    def _write_library(self, lib: Dict[str, Any]) -> None:
        with locked(self.library_path + ".lock"):
            atomic_write(self.library_path, lib)

    def _update_library(self, fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        with locked(self.library_path + ".lock"):
            lib = self._read_library()
            fn(lib)
            atomic_write(self.library_path, lib)
            return lib

    # -- audio-file registry --------------------------------------------- #

    def list_files(self) -> List[Dict[str, Any]]:
        lib = self._read_library()
        files = list(lib.get("files", {}).values())
        files.sort(key=lambda f: f.get("uploaded_at", ""), reverse=True)
        return files

    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        return self._read_library().get("files", {}).get(file_id)

    def add_file(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        file_id = entry.get("id") or new_id()
        entry.setdefault("id", file_id)
        entry.setdefault("uploaded_at", now_iso())
        entry.setdefault("derived_from", None)
        entry.setdefault("version", 1)
        entry.setdefault("tags", [])
        entry.setdefault("analyses", [])

        def _add(lib: Dict[str, Any]) -> None:
            lib.setdefault("files", {})[file_id] = entry

        self._update_library(_add)
        return entry

    def update_file(self, file_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        def _patch(lib: Dict[str, Any]) -> None:
            entry = lib.get("files", {}).get(file_id)
            if entry is None:
                return
            entry.update(patch)
            entry["updated_at"] = now_iso()

        self._update_library(_patch)
        return self.get_file(file_id)

    def delete_file(self, file_id: str) -> bool:
        """Delete an audio file and every derived artefact, keeping the index in sync."""
        def _delete(lib: Dict[str, Any]) -> None:
            entry = lib.get("files", {}).pop(file_id, None)
            if entry is None:
                return
            # Remove the audio blob.
            audio_path = os.path.join(self.root, entry.get("path", ""))
            if os.path.isfile(audio_path):
                os.unlink(audio_path)
            # Remove analyses (best-effort glob).
            for kind in entry.get("analyses", []):
                ap = self._analysis_path(file_id, kind)
                if os.path.isfile(ap):
                    os.unlink(ap)
            # Remove version snapshots for this file.
            for vp in self._version_glob(file_id):
                if os.path.isfile(vp):
                    os.unlink(vp)

        before = self.get_file(file_id)
        if before is None:
            return False
        with locked(self.library_path + ".lock"):
            lib = self._read_library()
            _delete(lib)
            atomic_write(self.library_path, lib)
        return True

    # -- analysis results ------------------------------------------------- #

    def _analysis_path(self, file_id: str, kind: str) -> str:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in kind)
        return os.path.join(self.analysis_dir, f"{file_id}__{safe}.json")

    def save_analysis(self, file_id: str, kind: str, data: Dict[str, Any],
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = {
            "file_id": file_id,
            "kind": kind,
            "created_at": now_iso(),
            "params": params or {},
            "data": data,
        }
        with locked(self._analysis_path(file_id, kind) + ".lock"):
            atomic_write(self._analysis_path(file_id, kind), doc)

        def _record(lib: Dict[str, Any]) -> None:
            entry = lib.get("files", {}).get(file_id)
            if entry is None:
                return
            kinds = entry.setdefault("analyses", [])
            if kind not in kinds:
                kinds.append(kind)

        self._update_library(_record)
        return doc

    def get_analysis(self, file_id: str, kind: str) -> Optional[Dict[str, Any]]:
        return read_json(self._analysis_path(file_id, kind), None)

    # -- projects --------------------------------------------------------- #

    def _project_path(self, project_id: str) -> str:
        return os.path.join(self.projects_dir, f"{project_id}.json")

    def create_project(self, name: str, tracks: Optional[List[Dict]] = None) -> Dict[str, Any]:
        project = {
            "id": new_id(),
            "name": name,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "version": 1,
            "tracks": tracks or [],
            "master": {"gain": 1.0},
        }
        with locked(self._project_path(project["id"]) + ".lock"):
            atomic_write(self._project_path(project["id"]), project)
        self._snapshot_project(project)
        return project

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return read_json(self._project_path(project_id), None)

    def list_projects(self) -> List[Dict[str, Any]]:
        out = []
        for name in os.listdir(self.projects_dir):
            if name.endswith(".json"):
                p = read_json(os.path.join(self.projects_dir, name), None)
                if p:
                    out.append(p)
        out.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
        return out

    def update_project(self, project_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = self._project_path(project_id)
        with locked(path + ".lock"):
            project = read_json(path, None)
            if project is None:
                return None
            project.update(patch)
            project["version"] = int(project.get("version", 0)) + 1
            project["updated_at"] = now_iso()
            atomic_write(path, project)
        self._snapshot_project(project)
        return project

    def delete_project(self, project_id: str) -> bool:
        path = self._project_path(project_id)
        if not os.path.isfile(path):
            return False
        with locked(path + ".lock"):
            os.unlink(path)
        return True

    # -- version history -------------------------------------------------- #

    def _version_glob(self, target_id: str) -> List[str]:
        prefix = f"{target_id}__"
        return [os.path.join(self.versions_dir, n) for n in os.listdir(self.versions_dir)
                if n.startswith(prefix) and n.endswith(".json")]

    def _snapshot_project(self, project: Dict[str, Any]) -> None:
        vid = f"{project['id']}__{int(project.get('version', 1)):04d}.json"
        snap = dict(project)
        snap["snapshot_at"] = now_iso()
        with locked(os.path.join(self.versions_dir, vid) + ".lock"):
            atomic_write(os.path.join(self.versions_dir, vid), snap)

    def list_versions(self, project_id: str) -> List[Dict[str, Any]]:
        out = []
        for vp in self._version_glob(project_id):
            v = read_json(vp, None)
            if v:
                out.append(v)
        out.sort(key=lambda v: v.get("version", 0))
        return out

    def get_version(self, project_id: str, version: int) -> Optional[Dict[str, Any]]:
        vp = os.path.join(self.versions_dir, f"{project_id}__{int(version):04d}.json")
        return read_json(vp, None)

    def revert_project(self, project_id: str, version: int) -> Optional[Dict[str, Any]]:
        snap = self.get_version(project_id, version)
        if snap is None:
            return None
        path = self._project_path(project_id)
        with locked(path + ".lock"):
            current = read_json(path, None)
            if current is None:
                return None
            restored = dict(snap)
            restored.pop("snapshot_at", None)
            restored["version"] = int(current.get("version", 0)) + 1
            restored["updated_at"] = now_iso()
            restored["reverted_from"] = int(current.get("version", 0))
            atomic_write(path, restored)
        self._snapshot_project(restored)
        return restored

    # -- stats ------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        lib = self._read_library()
        files = lib.get("files", {})
        total_bytes = sum(f.get("size_bytes", 0) for f in files.values())
        analyses = [n for n in os.listdir(self.analysis_dir) if n.endswith(".json")]
        projects = [n for n in os.listdir(self.projects_dir) if n.endswith(".json")]
        versions = [n for n in os.listdir(self.versions_dir) if n.endswith(".json")]
        return {
            "files": len(files),
            "total_bytes": total_bytes,
            "analyses": len(analyses),
            "projects": len(projects),
            "versions": len(versions),
            "has_ffmpeg": bool(__import__("shutil").which("ffmpeg")),
        }
