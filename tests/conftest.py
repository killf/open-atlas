import pytest
import torch


@pytest.fixture
def codec_dir(tmp_path):
    from open_atlas.codec import ImageCodec

    # 单元测试使用固定随机权重，不依赖网络，也不用于评价图像质量。
    torch.set_num_threads(2)
    torch.manual_seed(13)
    codec = ImageCodec()
    directory = tmp_path / "codec"
    directory.mkdir()
    for name in ["encoder", "decoder"]:
        torch.save(getattr(codec, name).state_dict(), directory / f"taesd_{name}.pth")
    return directory
