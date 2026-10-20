# SPDX-License-Identifier: Apache-2.0
"""Input construction and numerical checks shared by native GDN MTP cases."""

from types import SimpleNamespace

import torch

from ._gdn_mtp_reference import gdn_mtp_reference

DIM = 128


def make_inputs(
    lengths,
    *,
    head_ratio=2,
    key_heads=1,
    state_dtype=torch.float32,
    norm_dtype=torch.float32,
    bias_dtype=torch.float32,
    activation="silu",
    layout="packed",
    seed=0,
    width=None,
    permuted=True,
    indices=None,
    scale=DIM**-0.5,
    eps=1e-6,
    device="cuda",
):
    torch.manual_seed(seed)
    width = width if width is not None else max(lengths)
    tokens, requests = sum(lengths), len(lengths)
    heads = key_heads * head_ratio
    qkv_width = 2 * key_heads * DIM + heads * DIM
    mixed = torch.randn(tokens, qkv_width, dtype=torch.bfloat16, device=device)
    ba = torch.randn(tokens, 2 * heads, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    gate = torch.randn(tokens, heads, DIM, dtype=torch.bfloat16, device=device)
    if layout == "contiguous":
        a, b = a.contiguous(), b.contiguous()
    elif layout == "padded_rows":
        mixed_storage = torch.zeros(
            2 * tokens, qkv_width + 16, dtype=mixed.dtype, device=device
        )
        mixed_storage[::2, :qkv_width].copy_(mixed)
        mixed = mixed_storage[::2, :qkv_width]
        ba_storage = torch.zeros(2 * tokens, 2 * heads, dtype=ba.dtype, device=device)
        ba_storage[::2].copy_(ba)
        b, a = ba_storage[::2].chunk(2, dim=-1)
        gate_storage = torch.zeros(
            2 * tokens, heads, DIM, dtype=gate.dtype, device=device
        )
        gate_storage[::2].copy_(gate)
        gate = gate_storage[::2]
    else:
        assert layout in {"packed", "gapped_state"}

    num_slots = requests * width + 5  # NULL plus four unused guard slots.
    slot_stride = 2 if layout == "gapped_state" else 1
    backing = (
        0.01 * torch.randn(num_slots * slot_stride, heads, DIM, DIM, device=device)
    ).to(state_dtype)
    if indices is None:
        slot_ids = (
            torch.randperm(num_slots - 1, device=device)[: requests * width] + 1
            if permuted
            else torch.arange(1, requests * width + 1, device=device)
        )
        indices = slot_ids.to(torch.int32).view(requests, width)
        for request, length in enumerate(lengths):
            if length == 0:
                indices[request].zero_()

    kwargs = dict(
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=0.5 * torch.randn(heads, device=device),
        dt_bias=(0.1 * torch.randn(heads, device=device)).to(bias_dtype),
        state_indices=indices,
        cu_seqlens=torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        ),
        num_accepted_tokens=torch.ones(requests, dtype=torch.int32, device=device),
        output_gate=gate,
        norm_weight=torch.randn(DIM, dtype=norm_dtype, device=device),
        scale=scale,
        norm_eps=eps,
        output_gate_activation=activation,
    )
    return SimpleNamespace(
        kwargs=kwargs,
        lengths=tuple(lengths),
        backing=backing,
        state=backing[::slot_stride],
        slot_stride=slot_stride,
        width=width,
    )


def apply_signal_profile(inp, profile):
    kw = inp.kwargs
    device = kw["a"].device
    heads = kw["A_log"].numel()
    key_width = (kw["mixed_qkv"].shape[1] - heads * DIM) // 2
    if profile == "zero_q":
        kw["mixed_qkv"][:, :key_width].zero_()
    elif profile == "zero_k":
        kw["mixed_qkv"][:, key_width : 2 * key_width].zero_()
    elif profile in {"tiny_qk", "large_qk"}:
        kw["mixed_qkv"][:, : 2 * key_width].mul_(1e-4 if profile == "tiny_qk" else 100)
    elif profile == "zero_gate":
        kw["output_gate"].zero_()
    elif profile == "softplus_threshold":
        values = torch.tensor([-21, 19.875, 20, 20.125, 30], device=device)
        kw["a"].copy_(
            values[torch.arange(kw["a"].numel(), device=device) % 5].view_as(kw["a"])
        )
        kw["dt_bias"].zero_()
        kw["A_log"].fill_(-2)
    elif profile == "strong_decay":
        kw["A_log"].fill_(8)
        kw["a"].zero_()
        kw["dt_bias"].zero_()
    elif profile == "sigmoid_saturation":
        signs = torch.arange(heads, device=device) % 2 * 2 - 1
        kw["b"].copy_(signs * 40)
        kw["output_gate"].copy_(signs[None, :, None] * 40)
    else:
        raise ValueError(profile)


def initial_states(inp, *, eager=False):
    backing = inp.backing.clone()
    return SimpleNamespace(
        backing=backing,
        actual=backing[:: inp.slot_stride],
        triton=inp.state.clone(),
        eager=inp.state.cpu().clone() if eager else None,
    )


def _run_triton(runtime, kw, state):
    heads, key_dim, value_dim = state.shape[1], state.shape[3], state.shape[2]
    key_heads = (kw["mixed_qkv"].shape[1] - heads * value_dim) // (2 * key_dim)
    tokens = kw["mixed_qkv"].shape[0]
    q, k, v = kw["mixed_qkv"].split(
        [key_heads * key_dim, key_heads * key_dim, heads * value_dim], dim=-1
    )
    raw, _ = runtime.update(
        A_log=kw["A_log"],
        a=kw["a"],
        b=kw["b"],
        dt_bias=kw["dt_bias"],
        q=q.reshape(1, tokens, key_heads, key_dim),
        k=k.reshape(1, tokens, key_heads, key_dim),
        v=v.reshape(1, tokens, heads, value_dim),
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=kw["cu_seqlens"],
        ssm_state_indices=kw["state_indices"],
        num_accepted_tokens=kw["num_accepted_tokens"],
        scale=kw["scale"],
        use_qk_l2norm_in_kernel=True,
    )
    return runtime.rmsnorm(
        raw.squeeze(0),
        kw["norm_weight"],
        None,
        z=kw["output_gate"],
        eps=kw["norm_eps"],
        norm_before_gate=True,
        activation=kw["output_gate_activation"],
    )


def _token_and_write_masks(inp):
    kw = inp.kwargs
    indices = kw["state_indices"].tolist()
    accepted = kw["num_accepted_tokens"].tolist()
    active_tokens, written = [], set()
    for length, slots, count in zip(inp.lengths, indices, accepted, strict=False):
        active = slots[count - 1] > 0
        active_tokens.extend([active] * length)
        if active:
            written.update(slot for slot in slots[:length] if slot > 0)
    return torch.tensor(active_tokens, device="cuda", dtype=torch.bool), sorted(written)


def _assert_numerical(actual, expected, label, record_property, *, relative_limit=None):
    assert torch.isfinite(actual).all(), f"{label}: native contains NaN/Inf"
    assert torch.isfinite(expected).all(), f"{label}: reference contains NaN/Inf"
    if actual.numel() == 0:
        return
    difference = actual.float() - expected.float()
    relative = difference.norm() / expected.float().norm().clamp_min(1e-20)
    maximum = difference.abs().max()
    record_property(f"{label}_relative_l2", relative.item())
    record_property(f"{label}_max_abs", maximum.item())
    # Retain upstream's pointwise model-path bounds; the kernel output also
    # keeps upstream's stricter 5e-4 relative-L2 criterion.
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2, msg=label)
    if relative_limit is not None:
        assert relative < relative_limit, (
            f"{label}: relative L2={relative.item():.6g}, max abs={maximum.item():.6g}"
        )


def assert_unchanged(actual, expected, label):
    """Compare stored bits, including signed zero, rather than float equality."""
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), label


def check_round(inp, states, runtime, record_property, *, step=0, layer_entry=False):
    kw = inp.kwargs
    before, backing_before = states.actual.clone(), states.backing.clone()
    expected = _run_triton(runtime, kw, states.triton)
    guarded = torch.full(
        (kw["output_gate"].shape[0] + 2, *kw["output_gate"].shape[1:]),
        42,
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = guarded[1:-1]
    out.fill_(float("nan"))
    if layer_entry:
        layer = SimpleNamespace(
            A_log=kw["A_log"],
            dt_bias=kw["dt_bias"],
            kv_cache=(None, states.actual),
            norm=SimpleNamespace(
                weight=kw["norm_weight"], activation=kw["output_gate_activation"]
            ),
            head_k_dim=DIM,
            layer_norm_epsilon=kw["norm_eps"],
        )
        metadata = SimpleNamespace(
            spec_state_indices_tensor=kw["state_indices"],
            spec_query_start_loc=kw["cu_seqlens"],
            num_accepted_tokens=kw["num_accepted_tokens"],
            num_spec_decodes=len(inp.lengths),
        )
        runtime.layer._forward_core_decode_spec_post_conv_fused_norm(
            layer,
            mixed_qkv=kw["mixed_qkv"],
            b=kw["b"],
            a=kw["a"],
            output_gate=kw["output_gate"],
            core_attn_out=out,
            attn_metadata=metadata,
        )
        actual = out
    else:
        actual = runtime.ops.fused_gdn_decode_post_conv_mtp(
            **kw, state=states.actual, out=out
        )
        assert actual.data_ptr() == out.data_ptr()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all(), "native did not write every output token"
    assert_unchanged(
        guarded[[0, -1]], torch.full_like(guarded[[0, -1]], 42), "output guard changed"
    )

    active, written = _token_and_write_masks(inp)
    _assert_numerical(
        actual[active],
        expected[active],
        f"step{step}_output",
        record_property,
        relative_limit=5e-4,
    )
    assert torch.count_nonzero(actual[~active]) == 0, "NULL source output must be zero"
    _assert_numerical(
        states.actual, states.triton, f"step{step}_state", record_property
    )
    # Each updated slot must agree independently: a large untouched pool cannot
    # hide a wrong per-timestep write in an aggregate metric.
    for slot in written:
        _assert_numerical(
            states.actual[slot],
            states.triton[slot],
            f"step{step}_slot{slot}",
            record_property,
            relative_limit=3e-2,
        )
    untouched = torch.ones(states.actual.shape[0], dtype=torch.bool, device="cuda")
    untouched[written] = False
    assert_unchanged(states.actual[untouched], before[untouched], "unused slot changed")
    if inp.slot_stride == 2:
        assert_unchanged(
            states.backing[1::2], backing_before[1::2], "state gap changed"
        )

    if states.eager is not None:
        cpu_kw = {
            name: value.cpu() if isinstance(value, torch.Tensor) else value
            for name, value in kw.items()
        }
        eager_output = gdn_mtp_reference(**cpu_kw, state=states.eager)
        _assert_numerical(
            actual.cpu(), eager_output, f"step{step}_eager_output", record_property
        )
        _assert_numerical(
            states.actual.cpu(),
            states.eager,
            f"step{step}_eager_state",
            record_property,
        )


def check_case(inp, runtime, record_property, *, eager=False, layer_entry=False):
    states = initial_states(inp, eager=eager)
    counts = (
        [1] * len(inp.lengths),
        [
            max(1, min(length, request + 2))
            for request, length in enumerate(inp.lengths)
        ],
        [max(1, length) for length in inp.lengths],
    )
    for step, accepted in enumerate(counts):
        inp.kwargs["num_accepted_tokens"].copy_(torch.tensor(accepted, device="cuda"))
        check_round(
            inp, states, runtime, record_property, step=step, layer_entry=layer_entry
        )
