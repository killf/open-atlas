import torch


def test_euler_sampling_integrates_velocity():
    from open_atlas.model import sample_flow

    class ConstantVelocity(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays):
            return torch.ones_like(x) * 0.25

    initial = torch.zeros(1, 4, 8, 8)
    output = sample_flow(ConstantVelocity(), torch.zeros(1, 1, 3, 8, 8),
                         torch.zeros(1, 1, 6, 8, 8), torch.zeros(1, 6, 8, 8),
                         steps=4, noise=initial)
    torch.testing.assert_close(output, torch.ones_like(output) * 0.25)
    generated = sample_flow(ConstantVelocity(), torch.zeros(1, 1, 3, 8, 8),
                            torch.zeros(1, 1, 6, 8, 8), torch.zeros(1, 6, 8, 8), steps=1)
    assert generated.shape == (1, 4, 8, 8)


def test_small_rgbd_model_learns_and_uses_camera_condition():
    from open_atlas.model import AtlasConfig, AtlasMini

    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = AtlasMini(AtlasConfig(image_size=8, patch_size=4, dim=32, layers=1, heads=4))
    x = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 2, 3, 8, 8)
    context_rays = torch.randn(2, 2, 6, 8, 8)
    target_rays = torch.randn(2, 6, 8, 8)
    t = torch.ones(2) * 0.5
    before = model(x, t, context, context_rays, target_rays)
    changed = model(x, t, context, context_rays, target_rays + 1)
    assert before.shape == x.shape
    assert not torch.allclose(before, changed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    losses = []
    for _ in range(12):
        loss = model(x, t, context, context_rays, target_rays).square().mean()
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_flow_loss_does_not_learn_invalid_depth():
    from open_atlas.model import flow_loss

    class BadDepth(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays):
            prediction = torch.zeros_like(x)
            prediction[:, 3] = 100
            return prediction

    batch = {"target": torch.zeros(1, 4, 8, 8), "valid_depth": torch.zeros(1, 1, 8, 8),
             "context": torch.zeros(1, 1, 3, 8, 8), "context_rays": torch.zeros(1, 1, 6, 8, 8),
             "target_rays": torch.zeros(1, 6, 8, 8)}
    loss = flow_loss(BadDepth(), batch, noise=torch.zeros_like(batch["target"]), time=torch.ones(1) * .5)
    assert loss.item() == 0
