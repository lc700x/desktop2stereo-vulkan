import numpy as np
import pytest
from capture import capture_frame_to_rgb, prepare_rgb_for_depth_runtime
from capture.preprocess import (
    _mps_bt2020_to_bt709_gamut,
    _mps_p010_color_conversion_size,
    _prepare_mps_transfer_tensor,
)


def test_capture_frame_to_rgb_converts_bgr_numpy():
    frame = np.array([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    rgb = capture_frame_to_rgb(frame, target_height=2)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_capture_frame_to_rgb_converts_bgra_numpy():
    frame = np.array(
        [
            [[10, 20, 30, 255], [40, 50, 60, 128]],
            [[10, 20, 30, 255], [40, 50, 60, 128]],
        ],
        dtype=np.uint8,
    )

    rgb = capture_frame_to_rgb(frame, target_height=2)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_capture_frame_to_rgb_keeps_even_resize_dimensions():
    frame = np.zeros((3, 5, 3), dtype=np.uint8)

    rgb = capture_frame_to_rgb(frame, target_height=5)

    assert rgb.shape[0] % 2 == 0
    assert rgb.shape[1] % 2 == 0


def test_capture_frame_to_rgb_tensor_path_records_cpu_device_metadata():
    torch = pytest.importorskip("torch")
    frame = np.array([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    tensor = capture_frame_to_rgb(frame, target_height=2, device="cpu", output="tensor")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.device.type == "cpu"
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([30 / 255, 20 / 255, 10 / 255])
    assert tensor._d2s_preprocess_backend == "torch_bgr_norm"
    assert tensor._d2s_preprocess_input_kind == "numpy"
    assert tensor._d2s_preprocess_device_origin == "cpu"
    assert tensor._d2s_preprocess_device_output == "cpu"
    assert tensor._d2s_preprocess_device_transfer == "cpu->cpu"


def test_capture_frame_to_rgb_tensor_path_records_capture_metadata_overrides():
    torch = pytest.importorskip("torch")
    frame = np.array([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    tensor = capture_frame_to_rgb(
        frame,
        target_height=2,
        device="cpu",
        output="tensor",
        frame_raw_device="cuda",
        capture_copy_mode="clone",
        capture_zero_copy=False,
    )

    assert isinstance(tensor, torch.Tensor)
    assert tensor._d2s_preprocess_device_origin == "cuda"
    assert tensor._d2s_preprocess_device_output == "cpu"
    assert tensor._d2s_preprocess_device_transfer == "cuda->cpu"
    assert tensor._d2s_capture_copy_mode == "clone"
    assert tensor._d2s_capture_zero_copy is False


@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="requires MPS",
)
def test_mps_tensor_keeps_raw_cpu_capture_for_coreml_staging():
    torch = pytest.importorskip("torch")
    frame = np.zeros((4, 6, 4), dtype=np.uint8)

    tensor = capture_frame_to_rgb(frame, target_height=4, device="mps", output="tensor")

    raw = tensor._d2s_coreml_raw_bgr
    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == "cpu"
    assert tuple(raw.shape) == (4, 6, 4)


@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="requires MPS",
)
def test_prepare_rgb_for_stereo_runtime_stages_uint16_before_mps_transfer():
    torch = pytest.importorskip("torch")
    raw = torch.tensor(
        [
            [[0, 255], [512, 1023]],
            [[1023, 512], [255, 0]],
            [[128, 256], [768, 900]],
        ],
        dtype=torch.uint16,
    )

    prepared = prepare_rgb_for_depth_runtime(raw, device="mps")

    assert prepared.device.type == "mps"
    assert prepared.dtype == torch.float32
    expected = raw.float().div(255.0).clamp(0.0, 1.0).unsqueeze(0)
    torch.testing.assert_close(prepared.cpu(), expected)


def test_capture_frame_to_rgb_tensor_path_accepts_torch_hwc():
    torch = pytest.importorskip("torch")
    frame = torch.tensor([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=torch.uint8)

    tensor = capture_frame_to_rgb(frame, (2, 2), device="cpu", output="tensor")

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor._d2s_preprocess_input_kind == "torch.Tensor"
    assert tensor._d2s_preprocess_device_transfer == "cpu->cpu"


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "xpu", "hip"])
def test_same_torch_device_treats_default_device_as_device_zero(device_type):
    torch = pytest.importorskip("torch")
    from capture.preprocess import _same_torch_device

    assert _same_torch_device(torch.device(f"{device_type}:0"), torch.device(device_type), torch)
    assert not _same_torch_device(torch.device(f"{device_type}:1"), torch.device(device_type), torch)


def test_capture_frame_to_rgb_requires_one_target_resolution_argument():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Provide exactly one"):
        capture_frame_to_rgb(frame)
    with pytest.raises(TypeError, match="Provide exactly one"):
        capture_frame_to_rgb(frame, 2, target_height=2)


def test_prepare_rgb_for_depth_runtime_accepts_numpy_hwc():
    torch = pytest.importorskip("torch")
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)

    tensor = prepare_rgb_for_depth_runtime(frame, device="cpu")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 2, 3)
    assert float(tensor.max()) == 1.0


def test_prepare_rgb_for_depth_runtime_accepts_chw_and_bchw():
    torch = pytest.importorskip("torch")
    chw = torch.ones((3, 2, 4), dtype=torch.float32)
    bchw = torch.ones((1, 3, 2, 4), dtype=torch.float32)

    assert prepare_rgb_for_depth_runtime(chw, device="cpu").shape == (1, 3, 2, 4)
    assert prepare_rgb_for_depth_runtime(bchw, device="cpu").shape == (1, 3, 2, 4)


def test_prepare_rgb_for_depth_runtime_does_not_rescale_normalized_input():
    torch = pytest.importorskip("torch")
    frame = torch.full((3, 2, 2), 0.5, dtype=torch.float32)

    tensor = prepare_rgb_for_depth_runtime(frame, device="cpu")

    assert float(tensor.max()) == pytest.approx(0.5)


def test_capture_frame_to_rgb_tensor_path_tonemaps_hdr_bgr_like_rgbp():
    torch = pytest.importorskip("torch")
    rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    rgbp = np.transpose(rgb, (2, 0, 1)).copy()

    bgr_tensor = capture_frame_to_rgb(
        bgr,
        (2, 2),
        device="cpu",
        output="tensor",
        original_format="bgr24_pq_bt2020",
    )
    rgbp_tensor = capture_frame_to_rgb(
        rgbp,
        (2, 2),
        device="cpu",
        output="tensor",
        original_format="rgbp_pq_bt2020",
    )

    assert torch.allclose(bgr_tensor, rgbp_tensor)
    assert bgr_tensor._d2s_preprocess_backend == "torch_rgbp_hdr_resize_norm"


def test_pq_hable_uses_standard_1120_nit_white_point():
    pytest.importorskip("torch")
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    nits = 203.0
    normalized = nits / 10000.0
    encoded = ((c1 + c2 * normalized**m1) / (1.0 + c3 * normalized**m1)) ** m2
    code = round(64.0 + 876.0 * encoded)
    raw = np.empty((6, 4), dtype="<u2")
    raw[:4] = code << 6
    raw[4:] = 512 << 6

    frame = capture_frame_to_rgb(
        raw,
        (4, 4),
        device="cpu",
        output="tensor",
        original_format="p010le_pq_bt2020_bt2020nc_tv_center",
    )

    assert float(frame.mean()) == pytest.approx(0.734, abs=0.01)


@pytest.mark.parametrize("primaries", ["smpte432", "smpte431"])
def test_capture_frame_to_rgb_converts_p3_gamut_to_bt709(primaries):
    torch = pytest.importorskip("torch")
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :, 2] = 180

    tensor = capture_frame_to_rgb(
        bgr,
        (2, 2),
        device="cpu",
        output="tensor",
        original_format=f"bgr24_sdr_{primaries}",
    )

    # Preserve luminance while moving the negative BT.709 channels toward
    # neutral instead of clipping each channel independently.
    assert float(tensor[0, 0, 0, 0]) > 0.69
    assert torch.allclose(tensor[0, 0], tensor[0, 0, 0, 0].expand(2, 2))


@pytest.mark.parametrize(
    ("frame_format", "rgbp"),
    [
        ("bgr24_pq_bt2020", False),
        ("rgbp_pq_bt2020", True),
        ("bgr24_hlg_bt2020", False),
        ("rgbp_hlg_bt2020", True),
        ("bgr24_sdr_smpte432", False),
        ("rgbp_sdr_bt2020", True),
        ("rgbp", True),
    ],
)
def test_gpu_media_preprocess_matches_torch_reference(frame_format, rgbp):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")

    from capture.preprocess_triton import can_use_triton_media_preprocess

    height, width = 16, 32
    values = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    raw_cpu = np.transpose(values, (2, 0, 1)).copy() if rgbp else values[:, :, ::-1].copy()
    raw_gpu = torch.from_numpy(raw_cpu).to("cuda")
    if not can_use_triton_media_preprocess(raw_gpu):
        pytest.skip("Triton media preprocessing is unavailable")

    expected = capture_frame_to_rgb(
        raw_cpu,
        (width, height),
        device="cpu",
        output="tensor",
        original_format=frame_format,
    )
    actual = capture_frame_to_rgb(
        raw_gpu,
        (width, height),
        device="cuda",
        output="tensor",
        original_format=frame_format,
    )
    torch.cuda.synchronize()

    assert actual._d2s_preprocess_backend.startswith("triton_media_")
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)


def test_nv12_hdr_gpu_preprocess_matches_torch_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")

    height, width = 16, 32
    rng = np.random.default_rng(7)
    raw_cpu = rng.integers(16, 236, (height * 3 // 2, width), dtype=np.uint8)
    raw_cpu[height:, 1::2] = rng.integers(
        16, 241, (height // 2, width // 2), dtype=np.uint8
    )
    raw_cpu[height:, 0::2] = rng.integers(
        16, 241, (height // 2, width // 2), dtype=np.uint8
    )
    frame_format = "nv12_pq_bt2020_bt2020nc_tv_center"
    expected = capture_frame_to_rgb(
        raw_cpu,
        (width, height),
        device="cpu",
        output="tensor",
        original_format=frame_format,
    )
    actual = capture_frame_to_rgb(
        torch.from_numpy(raw_cpu).to("cuda"),
        (width, height),
        device="cuda",
        output="tensor",
        original_format=frame_format,
    )
    torch.cuda.synchronize()

    assert actual._d2s_preprocess_backend == "triton_media_nv12_hdr_fused"
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    ("target", "expected_backend"),
    [
        ((32, 16), "triton_media_p010_hdr_fused"),
        ((16, 8), "triton_media_p010_hdr_resize"),
    ],
)
def test_p010_hdr_gpu_preprocess_preserves_10_bit_samples(target, expected_backend):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")

    from capture.preprocess_triton import can_use_triton_media_preprocess

    height, width = 16, 32
    rng = np.random.default_rng(19)
    raw_cpu = np.empty((height * 3 // 2, width), dtype="<u2")
    raw_cpu[:height] = rng.integers(64, 941, (height, width), dtype=np.uint16) << 6
    raw_cpu[height:] = rng.integers(
        64, 961, (height // 2, width), dtype=np.uint16
    ) << 6
    raw_gpu = torch.from_numpy(raw_cpu).to("cuda")
    if not can_use_triton_media_preprocess(raw_gpu):
        pytest.skip("Triton media preprocessing is unavailable")

    frame_format = "p010le_pq_bt2020_bt2020nc_tv_center"
    expected = capture_frame_to_rgb(
        raw_cpu,
        target,
        device="cpu",
        output="tensor",
        original_format=frame_format,
    )
    actual = capture_frame_to_rgb(
        raw_gpu,
        target,
        device="cuda",
        output="tensor",
        original_format=frame_format,
    )
    torch.cuda.synchronize()

    assert actual._d2s_preprocess_backend == expected_backend
    torch.testing.assert_close(actual.cpu(), expected, rtol=4e-3, atol=4e-3)


def test_p010_hdr_cpu_preprocess_retains_adjacent_10_bit_levels():
    torch = pytest.importorskip("torch")
    frame_format = "p010le_pq_bt2020_bt2020nc_tv_center"
    first = np.full((6, 4), 0, dtype="<u2")
    second = np.full((6, 4), 0, dtype="<u2")
    first[:4] = 400 << 6
    second[:4] = 401 << 6
    first[4:] = 512 << 6
    second[4:] = 512 << 6

    first_rgb = capture_frame_to_rgb(
        first, (4, 4), device="cpu", output="tensor", original_format=frame_format
    )
    second_rgb = capture_frame_to_rgb(
        second, (4, 4), device="cpu", output="tensor", original_format=frame_format
    )

    assert first_rgb.dtype == torch.float32
    assert not torch.equal(first_rgb, second_rgb)


def test_p010_hdr_downsample_uses_area_filter_after_color_conversion():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    frame_format = "p010le_pq_bt2020_bt2020nc_tv_center"
    raw = np.empty((6, 4), dtype="<u2")
    raw[:4] = np.array(
        [[220, 360, 560, 780], [280, 430, 620, 820], [340, 490, 680, 860], [400, 540, 720, 900]],
        dtype=np.uint16,
    ) << 6
    raw[4:] = 512 << 6

    full = capture_frame_to_rgb(
        raw, (4, 4), device="cpu", output="tensor", original_format=frame_format
    )
    expected = F.interpolate(full, size=(2, 2), mode="area")
    actual = capture_frame_to_rgb(
        raw, (2, 2), device="cpu", output="tensor", original_format=frame_format
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_uint16_mps_transfer_stages_on_cpu_without_changing_samples():
    torch = pytest.importorskip("torch")
    raw = torch.tensor([0, 1, 1023, 65535], dtype=torch.uint16)

    prepared = _prepare_mps_transfer_tensor(raw, "mps", torch)

    assert prepared.device.type == "cpu"
    assert prepared.dtype == torch.int32
    torch.testing.assert_close(prepared, raw.to(dtype=torch.int32))


def test_mps_p010_hdr_color_grid_keeps_full_target_resolution(monkeypatch):
    monkeypatch.delenv("D2S_MEDIA_P010_MPS_COLOR_SCALE", raising=False)

    assert _mps_p010_color_conversion_size(4, 4) == (4, 4)
    assert _mps_p010_color_conversion_size(1920, 3840) == (1920, 3840)


def test_mps_p010_hdr_color_grid_ignores_legacy_downscale_setting(monkeypatch):
    monkeypatch.setenv("D2S_MEDIA_P010_MPS_COLOR_SCALE", "0.5")

    assert _mps_p010_color_conversion_size(1920, 3840) == (1920, 3840)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_mps_bt2020_gamut_transform_matches_matrix_reference(device):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    linear = torch.tensor(
        [
            [[0.0, 0.2, 1.0], [25.0, 3.0, 60.0]],
            [[0.5, 1.0, 0.0], [12.0, 40.0, 5.0]],
            [[1.0, 0.0, 0.4], [2.0, 8.0, 80.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    matrix = torch.tensor(
        (
            (1.660491, -0.587641, -0.072850),
            (-0.124550, 1.132900, -0.008349),
            (-0.018151, -0.100579, 1.118730),
        ),
        dtype=linear.dtype,
        device=linear.device,
    )
    converted = torch.matmul(matrix, linear.reshape(3, -1)).reshape_as(linear)
    luminance = (
        converted
        * torch.tensor(
            (0.2126, 0.7152, 0.0722), dtype=linear.dtype, device=linear.device
        ).view(3, 1, 1)
    ).sum(dim=0, keepdim=True)
    minimum = converted.min(dim=0, keepdim=True).values
    compression = torch.where(
        minimum < 0.0,
        (-minimum / (luminance - minimum).clamp_min(1e-6)).clamp(0.0, 1.0),
        torch.zeros_like(minimum),
    )
    expected = (converted + compression * (luminance - converted)).clamp_min(0.0)

    actual = _mps_bt2020_to_bt709_gamut(linear, torch)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=3e-5)
    if device == "mps":
        torch.mps.synchronize()


def test_p010_hdr_mps_preprocess_converts_uint16_before_device_transfer():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    frame_format = "p010le_pq_bt2020_bt2020nc_tv_center"
    raw = np.empty((6, 4), dtype="<u2")
    raw[:4] = 400 << 6
    raw[4:] = 512 << 6

    expected = capture_frame_to_rgb(
        raw, (4, 4), device="cpu", output="tensor", original_format=frame_format
    )
    actual = capture_frame_to_rgb(
        raw, (4, 4), device="mps", output="tensor", original_format=frame_format
    )

    assert actual.device.type == "mps"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="requires MPS",
)
def test_p010_hdr_mps_preprocess_antialias_resizes_to_requested_output():
    torch = pytest.importorskip("torch")
    frame_format = "p010le_pq_bt2020_bt2020nc_tv_center"
    raw = np.empty((6, 4), dtype="<u2")
    raw[:4] = np.array(
        [[220, 360, 560, 780], [280, 430, 620, 820], [340, 490, 680, 860], [400, 540, 720, 900]],
        dtype=np.uint16,
    ) << 6
    raw[4:] = 512 << 6

    expected = capture_frame_to_rgb(
        raw,
        (2, 2),
        device="cpu",
        output="tensor",
        original_format=frame_format,
    )
    actual = capture_frame_to_rgb(
        raw,
        (2, 2),
        device="mps",
        output="tensor",
        original_format=frame_format,
    )

    assert actual.device.type == "mps"
    assert tuple(actual.shape) == (1, 3, 2, 2)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="requires MPS",
)
def test_p010_hdr_mps_area_resize_supports_non_divisible_target_sizes():
    torch = pytest.importorskip("torch")
    frame_format = "p010le_pq_bt2020_bt2020nc_tv_center"
    raw = np.empty((9, 8), dtype="<u2")
    raw[:6] = np.array(
        [
            [180, 250, 340, 450, 570, 680, 780, 860],
            [210, 290, 380, 490, 600, 710, 800, 900],
            [250, 330, 420, 530, 640, 740, 830, 930],
            [290, 370, 460, 570, 670, 780, 870, 960],
            [330, 410, 500, 610, 710, 820, 910, 990],
            [370, 450, 540, 650, 750, 860, 950, 1000],
        ],
        dtype=np.uint16,
    ) << 6
    raw[6:] = 512 << 6

    expected = capture_frame_to_rgb(
        raw,
        (6, 4),
        device="cpu",
        output="tensor",
        original_format=frame_format,
    )
    actual = capture_frame_to_rgb(
        raw,
        (6, 4),
        device="mps",
        output="tensor",
        original_format=frame_format,
    )
    torch.mps.synchronize()

    assert actual.device.type == "mps"
    assert tuple(actual.shape) == (1, 3, 4, 6)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("frame_format", "rgbp"),
    [
        ("bgr24_pq_bt2020", False),
        ("rgbp_hlg_bt2020", True),
        ("bgr24_sdr_smpte432", False),
        ("rgbp_sdr_bt2020", True),
    ],
)
def test_gpu_media_preprocess_matches_torch_reference_when_resized(frame_format, rgbp):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")

    from capture.preprocess_triton import can_use_triton_media_preprocess

    height, width = 16, 32
    values = np.random.default_rng(43).integers(0, 256, (height, width, 3), dtype=np.uint8)
    raw_cpu = np.transpose(values, (2, 0, 1)).copy() if rgbp else values[:, :, ::-1].copy()
    raw_gpu = torch.from_numpy(raw_cpu).to("cuda")
    if not can_use_triton_media_preprocess(raw_gpu):
        pytest.skip("Triton media preprocessing is unavailable")

    target = (16, 8)
    expected = capture_frame_to_rgb(
        raw_cpu,
        target,
        device="cpu",
        output="tensor",
        original_format=frame_format,
    )
    actual = capture_frame_to_rgb(
        raw_gpu,
        target,
        device="cuda",
        output="tensor",
        original_format=frame_format,
    )
    torch.cuda.synchronize()

    assert actual.shape == (1, 3, 8, 16)
    assert actual._d2s_preprocess_backend in {
        "triton_media_hdr_resize",
        "triton_media_rgb_resize",
    }
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)


def test_nv12_hdr_gpu_preprocess_matches_torch_reference_when_resized():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")

    height, width = 16, 32
    rng = np.random.default_rng(8)
    raw_cpu = rng.integers(16, 236, (height * 3 // 2, width), dtype=np.uint8)
    raw_cpu[height:, 1::2] = rng.integers(
        16, 241, (height // 2, width // 2), dtype=np.uint8
    )
    raw_cpu[height:, 0::2] = rng.integers(
        16, 241, (height // 2, width // 2), dtype=np.uint8
    )
    frame_format = "nv12_pq_bt2020_bt2020nc_tv_center"
    expected = capture_frame_to_rgb(
        raw_cpu,
        (16, 8),
        device="cpu",
        output="tensor",
        original_format=frame_format,
    )
    actual = capture_frame_to_rgb(
        torch.from_numpy(raw_cpu).to("cuda"),
        (16, 8),
        device="cuda",
        output="tensor",
        original_format=frame_format,
    )
    torch.cuda.synchronize()

    assert actual.shape == (1, 3, 8, 16)
    assert actual._d2s_preprocess_backend == "triton_media_nv12_hdr_resize"
    torch.testing.assert_close(actual.cpu(), expected, rtol=4e-3, atol=4e-3)


def test_nv12_full_range_neutral_white_is_white():
    raw = np.full((6, 4), 128, dtype=np.uint8)
    raw[:4] = 255
    frame = capture_frame_to_rgb(
        raw,
        (4, 4),
        device="cpu",
        output="tensor",
        original_format="nv12_sdr_bt709_bt709_pc_center",
    )

    assert np.allclose(frame.numpy(), 1.0, atol=1e-5)


def test_gpu_preprocess_fuses_plain_bgr24_after_host_upload():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")

    from capture.preprocess_triton import can_use_triton_preprocess

    probe = torch.empty((2, 2, 3), device="cuda", dtype=torch.uint8)
    if not can_use_triton_preprocess(probe):
        pytest.skip("Triton preprocessing is unavailable")

    frame = np.random.default_rng(44).integers(0, 256, (32, 64, 3), dtype=np.uint8)
    expected = capture_frame_to_rgb(
        frame,
        target_height=16,
        device="cpu",
        output="tensor",
        original_format="bgr24",
    )
    actual = capture_frame_to_rgb(
        frame,
        target_height=16,
        device="cuda",
        output="tensor",
        original_format="bgr24",
    )
    torch.cuda.synchronize()

    assert actual.shape == (1, 3, 16, 32)
    assert actual._d2s_preprocess_backend == "triton_bgr_resize_norm"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
