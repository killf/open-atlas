import torch


def test_euler_sampling_preserves_unbounded_image_latents():
    from open_atlas.model import sample_flow

    class ConstantVelocity(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays, context_times):
            return torch.ones_like(x) * 2

    output = sample_flow(ConstantVelocity(), torch.zeros(1, 2, 4, 4, 4),
                         torch.zeros(1, 2, 6, 4, 4), torch.zeros(1, 6, 4, 4),
                         torch.tensor([[-2., -1.]]), steps=4, noise=torch.zeros(1, 4, 4, 4))
    torch.testing.assert_close(output, torch.full_like(output, 2))


def test_latent_model_learns_velocity_and_uses_camera_condition():
    from open_atlas.model import AtlasConfig, AtlasMini

    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = AtlasMini(AtlasConfig(latent_size=4, patch_size=1, dim=32, layers=1, heads=4))
    x = torch.randn(2, 4, 4, 4)
    context = torch.randn(2, 2, 4, 4, 4)
    context_rays = torch.randn(2, 2, 6, 4, 4)
    target_rays = torch.randn(2, 6, 4, 4)
    context_times = torch.tensor([[-2., -1.], [-2., -1.]])
    t = torch.full((2,), .5)
    target = torch.randn_like(x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    losses = []
    for _ in range(30):
        prediction = model(x, t, context, context_rays, target_rays, context_times)
        loss = (prediction - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * .5
    before = model(x, t, context, context_rays, target_rays, context_times)
    changed = model(x, t, context, context_rays, target_rays + 1, context_times)
    assert before.shape == x.shape
    assert not torch.allclose(before, changed)


def test_flow_loss_supervises_only_next_image_latent():
    from open_atlas.model import flow_loss

    class ZeroVelocity(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays, context_times):
            return torch.zeros_like(x)

    batch = {"target": torch.ones(1, 4, 4, 4), "context": torch.zeros(1, 2, 4, 4, 4),
             "context_rays": torch.zeros(1, 2, 6, 4, 4), "target_rays": torch.zeros(1, 6, 4, 4),
             "context_times": torch.tensor([[-2., -1.]])}
    loss = flow_loss(ZeroVelocity(), batch, noise=torch.zeros_like(batch["target"]), time=torch.full((1,), .5))
    assert loss.item() == 1
