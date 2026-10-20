"""Stream the official Replica v1.0 archive and retain only selected scenes."""

import argparse
import gzip
import hashlib
import http.client
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request


BASE = "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
URLS = [f"{BASE}/replica_v1_0.tar.gz.parta{c}" for c in "abcdefghijklmnopq"]
LICENSE_URL = "https://raw.githubusercontent.com/facebookresearch/Replica-Dataset/main/LICENSE"


class MultipartReader(io.RawIOBase):
    """按顺序拼接 HTTP 分卷，中断后从当前字节偏移恢复。"""

    def __init__(self, urls, timeout=60, retries=3, max_bytes=None):
        self.urls = list(urls)
        self.timeout, self.retries, self.max_bytes = timeout, retries, max_bytes
        self.index = self.offset = self.total = 0
        self.response = None
        self.size = None
        self.next_log = 256 * 2**20

    def readable(self):
        return True

    def _connect(self):
        req = urllib.request.Request(self.urls[self.index], headers={"Range": f"bytes={self.offset}-"})
        response = urllib.request.urlopen(req, timeout=self.timeout)
        status = getattr(response, "status", None)
        content_range = response.headers.get("Content-Range", "")
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != self.offset:
                response.close()
                raise ValueError("Server returned an incorrect Content-Range")
            size = int(match[3])
        elif not self.offset and response.headers.get("Content-Length"):
            size = int(response.headers["Content-Length"])
        else:
            response.close()
            # 续传时接受完整响应会重复拼入数据，因此必须拒绝。
            raise ValueError("Server ignored Range; refusing an unsafe resume")
        if self.size is not None and self.size != size:
            response.close()
            raise ValueError("Archive part changed size during download")
        self.size, self.response = size, response

    def readinto(self, buffer):
        if not len(buffer):
            return 0
        for attempt in range(self.retries + 1):
            if self.index >= len(self.urls):
                return 0
            try:
                if self.response is None:
                    self._connect()
                count = min(len(buffer), self.size - self.offset)
                if self.max_bytes is not None:
                    count = min(count, self.max_bytes - self.total)
                    if count <= 0:
                        raise ValueError("Download byte limit reached; increase --max-gb")
                try:
                    data = self.response.read(count)
                except http.client.IncompleteRead as exc:
                    data = exc.partial
                    self.response.close()
                    self.response = None
                if not data and self.offset < self.size:
                    raise OSError("Archive response ended before its Content-Length")
                self.offset += len(data)
                self.total += len(data)
                buffer[:len(data)] = data
                if self.total >= self.next_log:
                    print(f"Read {self.total / 2**30:.2f} GiB; archive part {self.index + 1}/17", flush=True)
                    self.next_log += 256 * 2**20
                if self.offset == self.size:
                    if self.response:
                        self.response.close()
                    self.response, self.size = None, None
                    self.index += 1
                    self.offset = 0
                return len(data)
            except (OSError, urllib.error.URLError, http.client.HTTPException):
                if self.response:
                    self.response.close()
                self.response = None
                if attempt == self.retries:
                    raise
        return 0

    def close(self):
        if self.response:
            self.response.close()
        super().close()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_manifest(output, manifest):
    path = Path(output) / "manifest.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(path)


def verified_file(output, name, entry):
    path = Path(output) / name
    return (not path.is_symlink() and path.is_file() and path.stat().st_size == entry["bytes"]
            and digest(path) == entry["sha256"])


def verify(output, scenes, geometry_only=False):
    try:
        manifest = json.loads((Path(output) / "manifest.json").read_text())
        for scene in scenes:
            record = manifest["scenes"][scene]
            if not record["complete"] or (not geometry_only and record["geometry_only"]):
                return False
            if f"{scene}/mesh.ply" not in record["files"]:
                return False
            if not all(verified_file(output, name, entry) for name, entry in record["files"].items()):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def extract_scenes(stream, output, scenes, geometry_only=False):
    """流式筛选场景，文件完整写入后才原子替换并记录校验值。"""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = set(scenes)
    if not scenes or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in scenes):
        raise ValueError("Scene names must be single directory names")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "dataset": "Replica v1.0", "source": BASE,
        "checksum_note": "Locally computed SHA256; upstream provides no trusted checksum manifest.",
        "scenes": {},
    }
    # 旧清单仅用于跳过已校验文件；完成状态必须依据本次实际读到的条目。
    previous_files = {}
    for scene in scenes:
        previous = manifest["scenes"].get(scene, {})
        previous_files[scene] = previous.get("files", {})
        manifest["scenes"][scene] = {"complete": False, "geometry_only": geometry_only,
                                     "files": {}}
    save_manifest(output, manifest)
    seen, finished = set(), set()
    active_scene = None
    with gzip.GzipFile(fileobj=stream) as decoded, tarfile.open(fileobj=decoded, mode="r|") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if not parts:
                continue
            if member.name.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            scene = parts[0]
            if active_scene in seen and scene != active_scene:
                finished.add(active_scene)
                if scenes <= finished:
                    break
            active_scene = scene
            if scene not in scenes:
                continue
            seen.add(scene)
            if member.issym() or member.islnk():
                raise ValueError(f"Archive link is not allowed: {member.name}")
            if not member.isfile() or (geometry_only and parts[1:] != ("mesh.ply",)):
                continue
            path = output.joinpath(*parts)
            if not path.resolve().is_relative_to(output):
                raise ValueError(f"Unsafe output path: {member.name}")
            name = "/".join(parts)
            record = manifest["scenes"][scene]
            old = previous_files[scene].get(name)
            if old and old["bytes"] == member.size and verified_file(output, name, old):
                record["files"][name] = old
                print(f"Verified existing {name}", flush=True)
            else:
                if shutil.disk_usage(output).free < member.size + 64 * 2**20:
                    raise OSError(f"Insufficient disk space for {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + ".part")
                if temp.is_symlink():
                    raise ValueError(f"Unsafe partial file path: {temp}")
                source = tar.extractfile(member)
                with source, temp.open("wb") as target:
                    shutil.copyfileobj(source, target, length=2**20)
                if temp.stat().st_size != member.size:
                    raise ValueError(f"Incomplete archive member: {name}")
                temp.replace(path)
                record["files"][name] = {"bytes": member.size, "sha256": digest(path)}
                save_manifest(output, manifest)
                print(f"Saved {name} ({member.size / 2**20:.1f} MiB)", flush=True)
            if geometry_only:
                finished.add(scene)
                if scenes <= finished:
                    break
        else:
            # tarfile 允许成员之后直接 EOF，仍须读取 gzip 尾部以检测传输截断。
            # 仅主动提前结束场景或网格提取时，可以不读取归档尾部。
            while decoded.read(2**20):
                pass
    for scene in scenes:
        record = manifest["scenes"][scene]
        if f"{scene}/mesh.ply" not in record["files"]:
            raise ValueError(f"Scene {scene} is missing mesh.ply")
        if not geometry_only and not any(n.startswith(f"{scene}/textures/") for n in record["files"]):
            raise ValueError(f"Scene {scene} is missing textures")
        record["complete"] = True
    save_manifest(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/replica"))
    parser.add_argument("--scenes", nargs="+", default=["apartment_0"])
    parser.add_argument("--geometry-only", action="store_true", help="Retain vertex-colored mesh only, discarding PTex")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Validate the official source with a 64 KiB range request")
    parser.add_argument("--max-gb", type=float, help="Optional limit on compressed bytes read")
    args = parser.parse_args()
    if args.probe:
        req = urllib.request.Request(URLS[0], headers={"Range": "bytes=0-65535"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read(65536)
            if response.status != 206 or not data.startswith(b"\x1f\x8b"):
                raise ValueError("Official source did not return a ranged gzip stream")
            print(json.dumps({"source": URLS[0], "status": response.status,
                              "content_range": response.headers["Content-Range"],
                              "bytes_read": len(data), "sha256": hashlib.sha256(data).hexdigest()}, indent=2))
        return
    if verify(args.output, args.scenes, args.geometry_only):
        print("Verified all requested scenes (file sizes and SHA256).")
        return
    if args.verify_only:
        raise ValueError("Scene verification failed: missing, incomplete or corrupt files")
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LICENSE_URL, timeout=60) as response:
        (args.output / "LICENSE").write_bytes(response.read())
    with MultipartReader(URLS, max_bytes=int(args.max_gb * 2**30) if args.max_gb else None) as stream:
        extract_scenes(stream, args.output, args.scenes, args.geometry_only)
        print(f"Read {stream.total:,} compressed bytes from the official archive.")
    if not verify(args.output, args.scenes, args.geometry_only):
        raise ValueError("Post-download verification failed")
    print("Download and SHA256 verification complete.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, tarfile.TarError, EOFError) as error:
        sys.exit(f"Error: {error}")
