import torch


def test_rgb_codec_compresses_and_is_frozen(codec_dir):
    from open_atlas.codec import ImageCodec

    codec = ImageCodec(codec_dir)
    rgb = torch.rand(2, 3, 16, 16)
    latent = codec.encode(rgb)
    assert latent.shape == (2, 4, 2, 2)
    result = codec.decode(latent)
    assert result.shape == rgb.shape
    assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    assert not any(p.requires_grad for p in codec.parameters())
