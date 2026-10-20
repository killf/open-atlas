import hashlib

import pytest


def test_codec_download_verifies_and_rejects_changed_source(tmp_path):
    from scripts.download_codec import download_file

    source = tmp_path / "source"
    source.write_bytes(b"official test weights")
    target = tmp_path / "weights"
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    download_file(source.as_uri(), target, checksum)
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        download_file(source.as_uri(), target, checksum, verify_only=True)
    source.write_bytes(b"changed upstream")
    with pytest.raises(ValueError, match="SHA256"):
        download_file(source.as_uri(), target, checksum)
    assert target.read_bytes() == b"corrupt"
