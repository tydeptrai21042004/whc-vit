import torch

from src.models.vit_adapter.dt1d_adapter import DT1DAdapter, DT1DTokenAdapter


def test_canonical_defaults_match_frozen_proposal():
    m = DT1DAdapter(32)
    assert m.proposal_name == "DT1D-Adapter"
    assert m.architecture_name == "R124-P2-G16-Axis-LearnedGate"
    assert m.axis == "hw"
    assert m.group_size == 16
    assert m.active_offsets == (1, 2, 4)
    assert m.detail_components == "offset4"
    assert m.shift_p == 2
    assert m.shift_lambda_mode == "learned"
    assert m.shift_lambda_scope == "axis"
    assert m.shift_lambda_max == 0.5
    assert m.gate_mode == "learned"
    assert isinstance(m.gate, torch.nn.Parameter)
    assert torch.isclose(m.gate.detach(), torch.tensor(0.01))
    assert m.project_l1 is True
    assert m.use_pointwise is False
    assert m.padding_mode == "replicate"
    assert m.base_kernel_size == 9
    assert m.effective_kernel_size == 13
    assert m.convolution_calls_per_forward == 2
    assert m.is_canonical_dt1d_adapter


def test_vit_b16_parameter_count_is_483_per_block():
    m = DT1DAdapter(768)
    b = m.parameter_count_breakdown()
    assert b["shared_base"] == 384
    assert b["channel_correction"] == 96
    assert b["shift_weight"] == 2
    assert b["learned_gate"] == 1
    assert b["pointwise"] == 0
    assert b["total"] == 483


def test_joint_l1_projection_caps_h_w_mass():
    torch.manual_seed(1)
    m = DT1DAdapter(32)
    with torch.no_grad():
        m.base_coefficients.normal_()
        m.detail_coefficients.normal_()
        m.shift_theta.normal_()
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    mass = k.squeeze(2).abs().sum(-1).sum(0)
    assert torch.all(mass <= 1.000001)


def test_axis_lambdas_are_independent_and_bounded():
    m = DT1DAdapter(16)
    assert m.shift_theta.shape == (2,)
    with torch.no_grad():
        m.shift_theta[:] = torch.tensor([10.0, -10.0])
    lam = m.shift_lambda(torch.device("cpu"), torch.float32)
    assert lam.shape == (2,)
    assert lam[0] > 0.49
    assert lam[1] < -0.49
    assert torch.all(lam.abs() <= 0.500001)


def test_zero_lambda_centers_k9_inside_k13():
    m = DT1DAdapter(16)
    base = m._build_base_kernel(torch.device("cpu"), torch.float32)
    full = m.build_unprojected_kernels(torch.device("cpu"), torch.float32).squeeze(2)
    assert base.shape[-1] == 9
    assert full.shape[-1] == 13
    assert torch.allclose(full[..., 2:11], base)
    assert torch.count_nonzero(full[..., :2]).item() == 0
    assert torch.count_nonzero(full[..., -2:]).item() == 0


def test_token_adapter_preserves_class_token_shape_and_gradients():
    torch.manual_seed(2)
    m = DT1DTokenAdapter(embed_dim=16, grid_size=(4, 4))
    x = torch.randn(2, 17, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :1], x[:, :1])
    y.square().mean().backward()
    s = m.spatial_adapter
    assert x.grad is not None
    assert s.shift_theta.grad is not None
    assert s.shift_theta.grad.abs().sum() > 0
    assert s.gate.grad is not None
    assert s.gate.grad.abs() > 0


def test_channel_contrast_is_zero_mean_per_full_group():
    m = DT1DAdapter(32, group_size=16)
    a = m.channel_contrast
    assert abs(float(a[:16].sum())) < 1e-6
    assert abs(float(a[16:32].sum())) < 1e-6


def test_reviewer_controls_do_not_change_public_method_name():
    m = DT1DAdapter(16, shift_lambda_mode="off", gate_mode="fixed")
    assert m.method_name == "DT1D-Adapter"
    assert m.proposal_name == "DT1D-Adapter"
    assert not m.is_canonical_dt1d_adapter


def test_removed_offset8_is_rejected():
    try:
        DT1DAdapter(16, active_offsets="1,2,4,8")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("offset 8 must not survive the canonical proposal cleanup")


def test_prefix_free_token_grid_is_supported():
    m = DT1DTokenAdapter(embed_dim=8, grid_size=(3, 3), num_prefix_tokens=0)
    x = torch.randn(2, 9, 8)
    assert m(x).shape == x.shape


def test_weighted_shift_matches_explicit_reference_formula():
    torch.manual_seed(7)
    m = DT1DAdapter(16, project_l1=False)
    with torch.no_grad():
        m.base_coefficients.normal_(mean=0.0, std=0.1)
        m.detail_coefficients.normal_(mean=0.0, std=0.1)
        m.shift_theta[:] = torch.tensor([0.3, -0.4])
    base = m._build_base_kernel(torch.device("cpu"), torch.float32)
    lam = m.shift_lambda(torch.device("cpu"), torch.float32).view(2, 1, 1)
    centered = torch.nn.functional.pad(base, (2, 2))
    shifted = torch.zeros_like(centered)
    shifted[..., :9] += base
    shifted[..., 4:13] += base
    shifted *= 0.5
    expected = ((1.0 - lam) * centered + lam * shifted).unsqueeze(2)
    got = m.build_unprojected_kernels(torch.device("cpu"), torch.float32)
    assert torch.allclose(got, expected, atol=1e-7, rtol=1e-6)


def test_inference_cache_matches_eager_and_is_cleared_by_train():
    torch.manual_seed(8)
    m = DT1DAdapter(16, cache_kernel=True).eval()
    x = torch.randn(2, 16, 5, 5)
    eager = m(x)
    m.prepare_for_inference()
    assert m._cached_kernels.numel() > 0
    cached = m(x)
    assert torch.allclose(eager, cached, atol=1e-7, rtol=1e-6)
    m.train()
    assert m._cached_kernels.numel() == 0


def test_state_dict_roundtrip_preserves_output():
    torch.manual_seed(9)
    a = DT1DAdapter(16)
    with torch.no_grad():
        for p in a.parameters():
            p.normal_(mean=0.0, std=0.05)
    x = torch.randn(1, 16, 6, 6)
    ya = a(x)
    b = DT1DAdapter(16)
    b.load_state_dict(a.state_dict())
    yb = b(x)
    assert torch.allclose(ya, yb, atol=1e-7, rtol=1e-6)


def test_fixed_and_off_shift_modes_do_not_create_trainable_lambda():
    fixed = DT1DAdapter(16, shift_lambda_mode="fixed", shift_lambda_init=0.25)
    off = DT1DAdapter(16, shift_lambda_mode="off")
    assert not hasattr(fixed, "shift_theta")
    assert not hasattr(off, "shift_theta")
    assert fixed.parameter_count_breakdown()["shift_weight"] == 0
    assert off.parameter_count_breakdown()["shift_weight"] == 0
    assert fixed.effective_kernel_size == 13
    assert off.effective_kernel_size == 9


def test_partial_channel_group_keeps_detail_contrast_active():
    m = DT1DAdapter(24, group_size=16, contrast_split=8)
    assert m.valid_contrast_group.tolist() == [1.0, 1.0]
    assert abs(float(m.channel_contrast[16:24].sum())) < 1e-6
    assert torch.count_nonzero(m.channel_contrast[16:24]).item() == 8


def test_non_square_patch_grid_preserves_prefix_token():
    m = DT1DTokenAdapter(embed_dim=8, grid_size=(2, 3), num_prefix_tokens=1)
    x = torch.randn(2, 7, 8)
    y = m(x)
    assert y.shape == x.shape
    assert torch.equal(y[:, 0], x[:, 0])
