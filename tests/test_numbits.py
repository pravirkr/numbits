import pytest
import numbits
import numpy as np


def unpack_bits(arr: np.ndarray, nbits: int, bitorder: str = "big") -> np.ndarray:
    assert arr.dtype == np.uint8
    assert nbits in {1, 2, 4}

    mask = (1 << nbits) - 1
    shifts = np.arange(0, 8, nbits, dtype=np.uint8)
    if bitorder == "big":
        shifts = shifts[::-1]
    unpacked = (arr[..., np.newaxis] >> shifts) & mask
    return unpacked.reshape(-1).astype(np.uint8)


def pack_bits(arr: np.ndarray, nbits: int, bitorder: str = "big") -> np.ndarray:
    assert arr.dtype == np.uint8
    assert nbits in {1, 2, 4}

    packed = np.zeros(arr.size * nbits // 8, dtype=np.uint8)
    shifts = np.arange(0, 8, nbits, dtype=np.uint8)
    if bitorder == "big":
        shifts = shifts[::-1]
    for ishift, shift in enumerate(shifts):
        # Cast to uint8 before shifting and after, to avoid upcasting
        vals = arr[ishift :: 8 // nbits].astype(np.uint8)
        packed |= (vals << shift).astype(np.uint8)
    return packed


class Testnumbits(object):
    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_lookup", [False, True])
    def test_unpack(self, nbits, bitorder, parallel, use_lookup):
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        expected = unpack_bits(arr, nbits, bitorder)
        output = numbits.unpack(
            arr,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_lookup=use_lookup,
        )
        np.testing.assert_array_equal(output, expected, strict=True)

    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_lookup", [False, True])
    def test_unpack_invalid(self, bitorder, parallel, use_lookup):
        arr = np.arange(255, dtype=np.uint8)
        with pytest.raises(ValueError):
            numbits.unpack(
                arr,
                nbits=3,
                bitorder=bitorder,
                parallel=parallel,
                use_lookup=use_lookup,
            )

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_lookup", [False, True])
    def test_unpack_empty(self, nbits, bitorder, parallel, use_lookup):
        arr = np.empty((0,), dtype=np.uint8)
        output = numbits.unpack(
            arr,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_lookup=use_lookup,
        )
        np.testing.assert_array_equal(output, arr, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_lookup", [False, True])
    def test_unpack_buffered(self, nbits, bitorder, parallel, use_lookup):
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        expected = unpack_bits(arr, nbits, bitorder)
        output = np.zeros(arr.size * 8 // nbits, dtype=np.uint8)
        numbits.unpack_buffered(
            arr,
            output,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_lookup=use_lookup,
        )
        np.testing.assert_array_equal(output, expected, strict=True)

    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_lookup", [False, True])
    def test_unpack_buffered_invalid(self, bitorder, parallel, use_lookup):
        arr = np.arange(255, dtype=np.uint8)
        output = np.zeros(arr.size * 8 // 3, dtype=np.uint8)
        with pytest.raises(ValueError):
            numbits.unpack_buffered(
                arr,
                output,
                nbits=3,
                bitorder=bitorder,
                parallel=parallel,
                use_lookup=use_lookup,
            )

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_lookup", [False, True])
    def test_unpack_buffered_empty(self, nbits, bitorder, parallel, use_lookup):
        arr = np.empty((0,), dtype=np.uint8)
        output = np.empty((0,), dtype=np.uint8)
        numbits.unpack_buffered(
            arr,
            output,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_lookup=use_lookup,
        )
        np.testing.assert_array_equal(output, arr, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_swar", [False, True])
    def test_pack(self, nbits, bitorder, parallel, use_swar):
        rng = np.random.default_rng()
        arr = rng.integers((1 << nbits) - 1, size=2**10, dtype=np.uint8)
        expected = pack_bits(arr, nbits, bitorder)
        output = numbits.pack(
            arr,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_swar=use_swar,
        )
        np.testing.assert_array_equal(output, expected, strict=True)

    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_swar", [False, True])
    def test_pack_invalid(self, bitorder, parallel, use_swar):
        arr = np.arange((1 << 3) - 1, dtype=np.uint8)
        with pytest.raises(ValueError):
            numbits.pack(
                arr,
                nbits=3,
                bitorder=bitorder,
                parallel=parallel,
                use_swar=use_swar,
            )

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_swar", [False, True])
    def test_pack_empty(self, nbits, bitorder, parallel, use_swar):
        arr = np.empty((0,), dtype=np.uint8)
        output = numbits.pack(
            arr,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_swar=use_swar,
        )
        np.testing.assert_array_equal(output, arr, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_swar", [False, True])
    def test_pack_buffered(self, nbits, bitorder, parallel, use_swar):
        rng = np.random.default_rng()
        arr = rng.integers((1 << nbits) - 1, size=2**10, dtype=np.uint8)
        expected = pack_bits(arr, nbits, bitorder)
        output = np.zeros(arr.size * nbits // 8, dtype=np.uint8)
        numbits.pack_buffered(
            arr,
            output,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_swar=use_swar,
        )
        np.testing.assert_array_equal(output, expected, strict=True)

    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_swar", [False, True])
    def test_pack_buffered_invalid(self, bitorder, parallel, use_swar):
        arr = np.arange((1 << 3) - 1, dtype=np.uint8)
        output = np.zeros(arr.size * 3 // 8, dtype=np.uint8)
        with pytest.raises(ValueError):
            numbits.pack_buffered(
                arr,
                output,
                nbits=3,
                bitorder=bitorder,
                parallel=parallel,
                use_swar=use_swar,
            )

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("use_swar", [False, True])
    def test_pack_buffered_empty(self, nbits, bitorder, parallel, use_swar):
        arr = np.empty((0,), dtype=np.uint8)
        output = np.empty((0,), dtype=np.uint8)
        numbits.pack_buffered(
            arr,
            output,
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
            use_swar=use_swar,
        )
        np.testing.assert_array_equal(output, arr, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_pack_unpack(self, nbits, bitorder, parallel):
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        output = numbits.pack(
            numbits.unpack(arr, nbits=nbits, bitorder=bitorder, parallel=parallel),
            nbits=nbits,
            bitorder=bitorder,
            parallel=parallel,
        )
        np.testing.assert_array_equal(output, arr, strict=True)

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    @pytest.mark.parametrize("bitorder", ["big", "little"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_pack_unpack_buffered(self, nbits, bitorder, parallel):
        rng = np.random.default_rng()
        arr = rng.integers(255, size=2**10, dtype=np.uint8)
        tmp = np.zeros(arr.size * 8 // nbits, dtype=np.uint8)
        numbits.unpack_buffered(
            arr, tmp, nbits=nbits, bitorder=bitorder, parallel=parallel
        )
        output = np.zeros_like(arr)
        numbits.pack_buffered(
            tmp, output, nbits=nbits, bitorder=bitorder, parallel=parallel
        )
        np.testing.assert_array_equal(output, arr, strict=True)
