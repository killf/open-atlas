import torch


def test_clean_image_prediction_integrates_to_rgb():
    from open_atlas.model import sample_flow

    class ConstantImage(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays, context_times):
            return torch.full_like(x, .7)

    context = torch.zeros(1, 2, 3, 4, 4)
    for steps in [1, 4, 20, 80]:
        output = sample_flow(ConstantImage(), context, torch.zeros(1, 2, 6, 4, 4),
                             torch.zeros(1, 6, 4, 4), torch.tensor([[-2., -1.]]),
                             steps=steps, noise=torch.randn(1, 3, 4, 4))
        torch.testing.assert_close(output, torch.full_like(output, .7))


def test_pixel_model_learns_clean_rgb_and_uses_camera_condition():
    from open_atlas.model import AtlasConfig, AtlasMini

    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = AtlasMini(AtlasConfig(image_size=8, patch_size=2, dim=32, layers=1, heads=4))
    x = torch.randn(2, 3, 8, 8)
    context = torch.rand(2, 2, 3, 8, 8)
    context_rays = torch.randn(2, 2, 6, 8, 8)
    target_rays = torch.randn(2, 6, 8, 8)
    context_times = torch.tensor([[-2., -1.], [-2., -1.]])
    t = torch.full((2,), .5)
    target = torch.rand_like(x)
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


def test_flow_loss_uses_clean_rgb_prediction_with_velocity_weighting():
    from open_atlas.model import flow_loss

    class ZeroImage(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays, context_times):
            return torch.zeros_like(x)

    batch = {"target": torch.ones(1, 3, 4, 4), "context": torch.zeros(1, 2, 3, 4, 4),
             "context_rays": torch.zeros(1, 2, 6, 4, 4), "target_rays": torch.zeros(1, 6, 4, 4),
             "context_times": torch.tensor([[-2., -1.]])}
    loss = flow_loss(ZeroImage(), batch, noise=torch.zeros_like(batch["target"]), time=torch.full((1,), .5))
    assert loss.item() == 4


def test_perfect_clean_rgb_has_zero_loss_near_clean_endpoint():
    from open_atlas.model import flow_loss

    target = torch.rand(3, 3, 4, 4)

    class PerfectImage(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays, context_times):
            return target

    batch = {"target": target, "context": torch.rand(3, 2, 3, 4, 4),
             "context_rays": torch.zeros(3, 2, 6, 4, 4), "target_rays": torch.zeros(3, 6, 4, 4),
             "context_times": torch.zeros(3, 2)}
    loss = flow_loss(PerfectImage(), batch, noise=torch.randn_like(target), time=torch.tensor([.5, .99, 1.]))
    assert torch.isfinite(loss) and loss.item() == 0


def test_default_flow_times_follow_reference_noise_distribution():
    from open_atlas.model import flow_loss

    class CaptureTimes(torch.nn.Module):
        def forward(self, x, t, context, context_rays, target_rays, context_times):
            self.times = t
            return torch.zeros_like(x)

    # JiT 的 logit-normal(-0.8, 0.8) 避免均匀采样与速度权重过度强调干净端。
    torch.manual_seed(17)
    count = 32768
    batch = {"target": torch.zeros(count, 3, 1, 1), "context": torch.zeros(count, 1, 3, 1, 1),
             "context_rays": torch.zeros(count, 1, 6, 1, 1), "target_rays": torch.zeros(count, 6, 1, 1),
             "context_times": torch.zeros(count, 1)}
    model = CaptureTimes()
    flow_loss(model, batch)
    logits = torch.logit(model.times)
    assert torch.isfinite(logits).all()
    assert abs(logits.mean().item() + .8) < .02
    assert abs(logits.std().item() - .8) < .02
    assert (model.times > .9).float().mean() < .001
