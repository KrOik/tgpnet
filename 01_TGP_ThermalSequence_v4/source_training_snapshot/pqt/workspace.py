from __future__ import annotations

import json
import os
import socket
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SLOTS = ("treeA", "treeB", "treeC", "treeD")


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def repository_root(code_root: Path | None = None) -> Path:
    cwd = Path(code_root or Path.cwd()).resolve()
    return Path(_git("rev-parse", "--show-toplevel", cwd=cwd)).resolve()


def primary_repository_root(code_root: Path | None = None) -> Path:
    checkout = repository_root(code_root)
    common_dir = Path(_git("rev-parse", "--git-common-dir", cwd=checkout))
    if not common_dir.is_absolute():
        common_dir = (checkout / common_dir).resolve()
    return common_dir.parent if common_dir.name == ".git" else checkout


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    primary_root: Path
    work_root: Path
    worktrees_root: Path
    locks_root: Path
    runs_root: Path

    @classmethod
    def discover(cls, code_root: Path | None = None) -> "WorkspaceLayout":
        primary = primary_repository_root(code_root)
        work = primary / "work"
        return cls(primary, work, work / "worktrees", work / "locks", work / "results")

    def checkout_name(self, code_root: Path | None = None) -> str:
        """Return the stable result namespace for a primary or managed checkout."""
        checkout = repository_root(code_root)
        if checkout.parent.resolve() == self.worktrees_root.resolve() and checkout.name in SLOTS:
            return checkout.name
        return "main"

    def slot_path(self, slot: str) -> Path:
        _validate_slot(slot)
        return self.worktrees_root / slot

    def lock_path(self, slot: str) -> Path:
        _validate_slot(slot)
        return self.locks_root / f"{slot}.json"

    @property
    def connectors_root(self) -> Path:
        return self.work_root / "connectors"

    def connector_path(self, connector_id: str) -> Path:
        if not connector_id or not connector_id.isalnum():
            raise ValueError("Connector ID must be a non-empty alphanumeric value.")
        return self.connectors_root / f"{connector_id}.json"


def _validate_slot(slot: str) -> None:
    if slot not in SLOTS:
        raise ValueError(f"Unknown workspace slot {slot!r}; expected one of {SLOTS}.")


def initialize_worktrees(layout: WorkspaceLayout) -> dict[str, str]:
    layout.worktrees_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    layout.locks_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    layout.runs_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: dict[str, str] = {}
    for slot in SLOTS:
        path = layout.slot_path(slot)
        branch = f"worktree/{slot}"
        if path.exists():
            actual_root = repository_root(path)
            if actual_root != path.resolve():
                raise RuntimeError(f"{path} exists but is not the expected worktree root.")
            result[slot] = "existing"
            continue
        branch_exists = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=layout.primary_root,
        ).returncode == 0
        command = ["git", "worktree", "add"]
        if not branch_exists:
            command.extend(["-b", branch])
        command.extend([str(path), branch if branch_exists else "main"])
        subprocess.run(command, cwd=layout.primary_root, check=True)
        result[slot] = "created"
    return result


def _slot_state(layout: WorkspaceLayout, slot: str) -> dict[str, object]:
    path = layout.slot_path(slot)
    lock_path = layout.lock_path(slot)
    payload: dict[str, object] = {"slot": slot, "path": str(path), "exists": path.exists()}
    if path.exists():
        payload["branch"] = _git("branch", "--show-current", cwd=path)
        payload["head"] = _git("rev-parse", "HEAD", cwd=path)
        payload["dirty"] = bool(_git("status", "--porcelain", cwd=path))
    payload["lock"] = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    return payload


def workspace_status(layout: WorkspaceLayout) -> list[dict[str, object]]:
    return [_slot_state(layout, slot) for slot in SLOTS]


def claim_workspace(layout: WorkspaceLayout, slot: str, owner: str, task: str, job_id: str | None = None) -> dict[str, object]:
    _validate_slot(slot)
    state = _slot_state(layout, slot)
    if not state["exists"]:
        raise RuntimeError(f"Workspace {slot} is not initialized; run 'tgpnet workspace init'.")
    if state.get("dirty"):
        raise RuntimeError(f"Workspace {slot} has uncommitted changes and cannot be claimed.")
    layout.locks_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = uuid.uuid4().hex
    payload: dict[str, object] = {
        "slot": slot,
        "owner": owner,
        "task": task,
        "token": token,
        "branch": state.get("branch"),
        "head": state.get("head"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "job_id": job_id,
        "claimed_at": datetime.now(timezone.utc).isoformat(),
    }
    lock_path = layout.lock_path(slot)
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def release_workspace(layout: WorkspaceLayout, slot: str, token: str) -> None:
    _validate_slot(slot)
    lock_path = layout.lock_path(slot)
    if not lock_path.exists():
        raise RuntimeError(f"Workspace {slot} is not claimed.")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("token") != token:
        raise PermissionError(f"Token does not own workspace {slot}.")
    if _slot_state(layout, slot).get("dirty"):
        raise RuntimeError(f"Workspace {slot} is dirty; commit or otherwise preserve the work before release.")
    job_id = payload.get("job_id")
    if job_id:
        check = subprocess.run(["squeue", "-h", "-j", str(job_id)], capture_output=True, text=True)
        if check.returncode == 0 and check.stdout.strip():
            raise RuntimeError(f"Slurm job {job_id} is still active; workspace {slot} remains locked.")
    lock_path.unlink()


def _owned_clean_state(layout: WorkspaceLayout, slot: str, token: str) -> dict[str, object]:
    state = _slot_state(layout, slot)
    if not state["exists"]:
        raise RuntimeError(f"Workspace {slot} is not initialized.")
    lock = state.get("lock")
    if not isinstance(lock, dict) or lock.get("token") != token:
        raise PermissionError(f"Token does not own workspace {slot}.")
    if state.get("dirty"):
        raise RuntimeError(f"Workspace {slot} is dirty; synchronization requires a clean checkout.")
    return state


def declare_connector(layout: WorkspaceLayout, source: str, target: str, source_token: str,
                      target_token: str, purpose: str) -> dict[str, object]:
    """Declare a one-use, directional synchronization boundary."""
    _validate_slot(source)
    _validate_slot(target)
    if source == target:
        raise ValueError("Connector source and target must be different workspaces.")
    if not purpose.strip():
        raise ValueError("Connector purpose must be explicit.")
    source_state = _owned_clean_state(layout, source, source_token)
    target_state = _owned_clean_state(layout, target, target_token)
    layout.connectors_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    connector_id = uuid.uuid4().hex
    payload: dict[str, object] = {
        "id": connector_id, "source": source, "target": target, "purpose": purpose,
        "source_branch": source_state["branch"], "target_branch": target_state["branch"],
        "declared_source_head": source_state["head"], "declared_target_head": target_state["head"],
        "status": "declared", "declared_at": datetime.now(timezone.utc).isoformat(),
    }
    path = layout.connector_path(connector_id)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def synchronize_connector(layout: WorkspaceLayout, connector_id: str, source_token: str,
                          target_token: str) -> dict[str, object]:
    """Merge the declared source into target without force/reset or conflict resolution."""
    path = layout.connector_path(connector_id)
    if not path.exists():
        raise RuntimeError(f"Connector {connector_id} does not exist.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "declared":
        raise RuntimeError(f"Connector {connector_id} has already been used.")
    source, target = str(payload["source"]), str(payload["target"])
    source_state = _owned_clean_state(layout, source, source_token)
    target_state = _owned_clean_state(layout, target, target_token)
    if source_state["branch"] != payload["source_branch"] or target_state["branch"] != payload["target_branch"]:
        raise RuntimeError("A connected workspace changed branch after declaration.")
    target_path = layout.slot_path(target)
    source_head, before = str(source_state["head"]), str(target_state["head"])
    result = subprocess.run(
        ["git", "merge", "--no-ff", "--no-edit", "-m",
         f"Sync {source} into {target}: {payload['purpose']}", source_head],
        cwd=target_path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        merge_head = subprocess.run(["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
                                    cwd=target_path, capture_output=True, text=True)
        if merge_head.returncode == 0:
            subprocess.run(["git", "merge", "--abort"], cwd=target_path, check=True)
        raise RuntimeError("Synchronization is incompatible and was not applied: " +
                           (result.stderr.strip() or result.stdout.strip()))
    payload.update({
        "status": "synchronized", "source_head": source_head, "target_head_before": before,
        "target_head_after": _git("rev-parse", "HEAD", cwd=target_path),
        "synchronized_at": datetime.now(timezone.utc).isoformat(),
    })
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
