import gzip
import io
import tarfile
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def archive(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return gzip.compress(buf.getvalue())


def test_extract_only_requested_scene_and_repair_corruption(tmp_path):
    from scripts.download_replica import extract_scenes, verify

    data = archive({"room_0/mesh.ply": b"ply\nmesh", "room_0/textures/a.hdr": b"color",
                    "office_0/mesh.ply": b"other"})
    extract_scenes(io.BytesIO(data), tmp_path, ["room_0"])
    assert (tmp_path / "room_0/mesh.ply").read_bytes() == b"ply\nmesh"
    assert not (tmp_path / "office_0").exists()
    assert verify(tmp_path, ["room_0"])
    (tmp_path / "room_0/mesh.ply").write_bytes(b"corrupt")
    assert not verify(tmp_path, ["room_0"])
    extract_scenes(io.BytesIO(data), tmp_path, ["room_0"])
    assert verify(tmp_path, ["room_0"])


def test_reject_path_traversal(tmp_path):
    from scripts.download_replica import extract_scenes

    data = archive({"room_0/../../escape": b"bad"})
    with pytest.raises(ValueError, match="path"):
        extract_scenes(io.BytesIO(data), tmp_path, ["room_0"])
    assert not (tmp_path.parent / "escape").exists()


def test_missing_mesh_does_not_complete_scene(tmp_path):
    from scripts.download_replica import extract_scenes, verify

    with pytest.raises(ValueError, match="mesh"):
        extract_scenes(io.BytesIO(archive({"room_0/textures/a.hdr": b"a"})),
                       tmp_path, ["room_0"])
    assert not verify(tmp_path, ["room_0"])


def test_geometry_only_and_split_gzip_stream(tmp_path):
    from scripts.download_replica import MultipartReader, extract_scenes, verify

    payload = archive({"room_0/textures/a.hdr": b"texture", "room_0/mesh.ply": b"ply\n"})
    urls = []
    for i, chunk in enumerate([payload[:25], payload[25:71], payload[71:]]):
        p = tmp_path / f"part{i}"
        p.write_bytes(chunk)
        urls.append(p.as_uri())
    out = tmp_path / "out"
    with MultipartReader(urls) as reader:
        extract_scenes(reader, out, ["room_0"], geometry_only=True)
    assert verify(out, ["room_0"], geometry_only=True)
    assert not verify(out, ["room_0"], geometry_only=False)
    assert not (out / "room_0/textures").exists()


@pytest.mark.parametrize("supports_range", [True, False])
def test_http_interruption_resumes_or_rejects_ignored_range(supports_range):
    from scripts.download_replica import MultipartReader

    payload = b"0123456789" * 100
    offsets = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            offsets.append(start)
            self.send_response(206 if supports_range else 200)
            if supports_range:
                self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[:37] if len(offsets) == 1 else payload[start:])
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with MultipartReader([f"http://127.0.0.1:{server.server_port}/part"]) as stream:
            if supports_range:
                assert stream.read() == payload
            else:
                with pytest.raises(ValueError, match="ignored Range"):
                    stream.read()
        assert offsets == [0, 37]
    finally:
        server.shutdown()
        server.server_close()


def test_reject_symlink_and_truncated_member(tmp_path):
    from scripts.download_replica import extract_scenes

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        entry = tarfile.TarInfo("room_0/mesh.ply")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "/etc/passwd"
        tar.addfile(entry)
    with pytest.raises(ValueError, match="link"):
        extract_scenes(io.BytesIO(gzip.compress(buf.getvalue())), tmp_path, ["room_0"])


def test_truncated_gzip_cannot_complete_a_scene(tmp_path):
    from scripts.download_replica import extract_scenes, verify

    # The first files are complete, but the transfer loses later files and the gzip trailer.
    compressor = zlib.compressobj(wbits=31)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name in ["room_0/mesh.ply", "room_0/textures/first.hdr"]:
            member = tarfile.TarInfo(name)
            member.size = 3
            tar.addfile(member, io.BytesIO(b"abc"))
    incomplete = compressor.compress(raw.getvalue()[:2048]) + compressor.flush(zlib.Z_SYNC_FLUSH)
    with pytest.raises((EOFError, tarfile.ReadError)):
        extract_scenes(io.BytesIO(incomplete), tmp_path, ["room_0"])
    assert not verify(tmp_path, ["room_0"])


def test_previous_manifest_cannot_supply_missing_archive_members(tmp_path):
    from scripts.download_replica import extract_scenes, verify

    extract_scenes(io.BytesIO(archive({"room_0/mesh.ply": b"mesh"})), tmp_path, ["room_0"], True)
    with pytest.raises(ValueError, match="mesh"):
        extract_scenes(io.BytesIO(archive({"room_0/textures/new.hdr": b"a"})), tmp_path, ["room_0"])
    assert not verify(tmp_path, ["room_0"])
