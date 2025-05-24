#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <stdexcept>
#include <string_view>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace numbits {

inline constexpr uint8_t LO4BITS = 15; // 0x0F
inline constexpr uint8_t LO2BITS = 3;  // 0x03

enum class BitOrder : bool { LittleEndian = false, BigEndian = true };

// Compile-time bit size traits
template <size_t NBits> struct BitTraits {
    static_assert(NBits == 1 || NBits == 2 || NBits == 4,
                  "NBits must be 1, 2, or 4");
};

template <> struct BitTraits<1> {
    using PackType                            = uint64_t;
    static constexpr size_t elements_per_byte = 8;
    // Mask to isolate the least significant bit of each byte
    static constexpr PackType mask = 0x0101010101010101ULL;
    // SWAR constant to position bits correctly
    static constexpr PackType swar_big = 0x8040201008040201ULL;
    static constexpr PackType swar_lit = 0x0102040810204080ULL;
    static constexpr size_t swar_shift = 56;
};

template <> struct BitTraits<2> {
    using PackType                            = uint32_t;
    static constexpr size_t elements_per_byte = 4;
    static constexpr PackType mask            = 0x03030303UL;
    static constexpr PackType swar_big        = 0x40100401UL;
    static constexpr PackType swar_lit        = 0x01041040UL;
    static constexpr size_t swar_shift        = 24;
};

template <> struct BitTraits<4> {
    using PackType                            = uint16_t;
    static constexpr size_t elements_per_byte = 2;
    static constexpr PackType mask            = 0x0F0FU;
    static constexpr PackType swar_big        = 0x1001U;
    static constexpr PackType swar_lit        = 0x0110U;
    static constexpr size_t swar_shift        = 8;
};

// Convert string to BitOrder at compile time when possible
[[nodiscard]] constexpr BitOrder parse_bit_order(std::string_view bitorder) {
    if (bitorder.empty()) {
        throw std::invalid_argument("Empty bitorder string");
    }
    switch (bitorder[0]) {
    case 'l':
    case 'L':
        return BitOrder::LittleEndian;
    case 'b':
    case 'B':
        return BitOrder::BigEndian;
    default:
        throw std::invalid_argument(
            "Invalid bitorder. Must begin with 'l' or 'b'.");
    }
}

/*----------------------------------------------------------------------------*/
// Lookup table for bit unpacking
template <size_t NBits, BitOrder Order> class UnpackLookupTable {
    using Traits   = BitTraits<NBits>;
    using PackType = typename Traits::PackType;

    static constexpr size_t Size              = 256;
    static constexpr size_t elements_per_byte = Traits::elements_per_byte;
    static constexpr bool IsBigEndian         = (Order == BitOrder::BigEndian);

    alignas(64) std::array<PackType, Size> data_{}; // 256 * 8/NBits bytes

    constexpr void initialize() noexcept {
        constexpr uint8_t element_mask = (1 << NBits) - 1;
        for (size_t i = 0; i < Size; ++i) {
            PackType value = 0;
            for (size_t j = 0; j < elements_per_byte; ++j) {
                const uint8_t bits = (i >> (j * NBits)) & element_mask;
                const size_t shift =
                    IsBigEndian ? ((elements_per_byte - 1 - j) * 8) : (j * 8);
                value |= static_cast<PackType>(bits) << shift;
            }
            data_[i] = value;
        }
    }

public:
    constexpr UnpackLookupTable() noexcept { initialize(); }
    [[nodiscard]] constexpr PackType operator[](size_t idx) const noexcept {
        return data_[idx];
    }
    [[nodiscard]] constexpr const PackType* data() const noexcept {
        return data_.data();
    }
};

template <size_t NBits, BitOrder Order>
inline constexpr UnpackLookupTable<NBits, Order> lookup_table{};

template <size_t NBits, BitOrder Order, bool Parallel>
void unpack_lookup_impl(const uint8_t* __restrict__ in,
                        uint8_t* __restrict__ out,
                        size_t nbytes) noexcept {
    using Traits   = BitTraits<NBits>;
    using PackType = typename Traits::PackType;

    constexpr auto& table                   = lookup_table<NBits, Order>;
    const PackType* __restrict__ table_data = table.data();
    auto* __restrict__ out_packed           = reinterpret_cast<PackType*>(out);

    // Process in chunks for better cache utilization
    constexpr size_t chunk_size = 8;
    const size_t main_loop      = nbytes & ~(chunk_size - 1);

#ifdef USE_OPENMP
#pragma omp parallel for if (Parallel)
#endif
    for (size_t i = 0; i < main_loop; i += chunk_size) {
        // Prefetch next cache line
        __builtin_prefetch(&in[i + chunk_size], 0, 1);

#if defined(__clang__)
#pragma clang loop unroll(full) vectorize(enable)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#pragma GCC ivdep
#endif
        for (size_t j = 0; j < chunk_size; ++j) {
            out_packed[i + j] = table_data[in[i + j]];
        }
    }

    // Handle remaining elements
    for (size_t i = main_loop; i < nbytes; ++i) {
        out_packed[i] = table_data[in[i]];
    }
}

template <size_t NBits, BitOrder Order, bool Parallel>
void unpack_compute_impl(const uint8_t* __restrict__ in,
                         uint8_t* __restrict__ out,
                         size_t nbytes) noexcept {
    constexpr bool IsBigEndian         = (Order == BitOrder::BigEndian);
    constexpr size_t elements_per_byte = 8 / NBits;
#ifdef USE_OPENMP
#pragma omp parallel for if (Parallel)
#endif
    for (size_t i = 0; i < nbytes; ++i) {
        const size_t out_idx = i * elements_per_byte;
        const uint8_t byte   = in[i];
        if constexpr (NBits == 1) {
            if constexpr (IsBigEndian) {
                out[out_idx + 7] = (byte >> 0) & 1;
                out[out_idx + 6] = (byte >> 1) & 1;
                out[out_idx + 5] = (byte >> 2) & 1;
                out[out_idx + 4] = (byte >> 3) & 1;
                out[out_idx + 3] = (byte >> 4) & 1;
                out[out_idx + 2] = (byte >> 5) & 1;
                out[out_idx + 1] = (byte >> 6) & 1;
                out[out_idx + 0] = (byte >> 7) & 1;
            } else {
                out[out_idx + 0] = (byte >> 0) & 1;
                out[out_idx + 1] = (byte >> 1) & 1;
                out[out_idx + 2] = (byte >> 2) & 1;
                out[out_idx + 3] = (byte >> 3) & 1;
                out[out_idx + 4] = (byte >> 4) & 1;
                out[out_idx + 5] = (byte >> 5) & 1;
                out[out_idx + 6] = (byte >> 6) & 1;
                out[out_idx + 7] = (byte >> 7) & 1;
            }
        } else if constexpr (NBits == 2) {
            if constexpr (IsBigEndian) {
                out[out_idx + 3] = byte & LO2BITS;
                out[out_idx + 2] = (byte >> 2) & LO2BITS;
                out[out_idx + 1] = (byte >> 4) & LO2BITS;
                out[out_idx + 0] = (byte >> 6) & LO2BITS;
            } else {
                out[out_idx + 0] = byte & LO2BITS;
                out[out_idx + 1] = (byte >> 2) & LO2BITS;
                out[out_idx + 2] = (byte >> 4) & LO2BITS;
                out[out_idx + 3] = (byte >> 6) & LO2BITS;
            }
        } else if constexpr (NBits == 4) {
            if constexpr (IsBigEndian) {
                out[out_idx + 1] = byte & LO4BITS;
                out[out_idx + 0] = byte >> 4;
            } else {
                out[out_idx + 0] = byte & LO4BITS;
                out[out_idx + 1] = byte >> 4;
            }
        }
    }
}

template <size_t NBits, BitOrder Order, bool Parallel>
void pack_compute_impl(const uint8_t* __restrict__ in,
                       uint8_t* __restrict__ out,
                       size_t nbytes) noexcept {
    constexpr bool IsBigEndian         = (Order == BitOrder::BigEndian);
    constexpr size_t elements_per_byte = 8 / NBits;
    const size_t out_bytes             = nbytes / elements_per_byte;

#ifdef USE_OPENMP
#pragma omp parallel for if (Parallel)
#endif
    for (size_t i = 0; i < out_bytes; ++i) {
        const size_t in_idx = i * elements_per_byte;
        uint8_t byte        = 0;

        if constexpr (NBits == 1) {
            if constexpr (IsBigEndian) {
                byte = (in[in_idx + 0] << 7) | (in[in_idx + 1] << 6) |
                       (in[in_idx + 2] << 5) | (in[in_idx + 3] << 4) |
                       (in[in_idx + 4] << 3) | (in[in_idx + 5] << 2) |
                       (in[in_idx + 6] << 1) | in[in_idx + 7];
            } else {
                byte = in[in_idx + 0] | (in[in_idx + 1] << 1) |
                       (in[in_idx + 2] << 2) | (in[in_idx + 3] << 3) |
                       (in[in_idx + 4] << 4) | (in[in_idx + 5] << 5) |
                       (in[in_idx + 6] << 6) | (in[in_idx + 7] << 7);
            }
        } else if constexpr (NBits == 2) {
            if constexpr (IsBigEndian) {
                byte = (in[in_idx + 0] << 6) | (in[in_idx + 1] << 4) |
                       (in[in_idx + 2] << 2) | in[in_idx + 3];
            } else {
                byte = in[in_idx + 0] | (in[in_idx + 1] << 2) |
                       (in[in_idx + 2] << 4) | (in[in_idx + 3] << 6);
            }
        } else if constexpr (NBits == 4) {
            if constexpr (IsBigEndian) {
                byte = (in[in_idx] << 4) | in[in_idx + 1];
            } else {
                byte = in[in_idx] | (in[in_idx + 1] << 4);
            }
        }

        out[i] = byte;
    }
}

template <size_t NBits, BitOrder Order, bool Parallel>
void pack_swar_impl(const uint8_t* __restrict__ in,
                    uint8_t* __restrict__ out,
                    size_t nbytes) noexcept {
    using Traits   = BitTraits<NBits>;
    using PackType = typename Traits::PackType;

    constexpr bool IsBigEndian = (Order == BitOrder::BigEndian);
    constexpr PackType mask    = Traits::mask;
    constexpr PackType swar = IsBigEndian ? Traits::swar_big : Traits::swar_lit;
    constexpr size_t shift  = Traits::swar_shift;
    constexpr size_t elements_per_byte = Traits::elements_per_byte;

    const size_t out_bytes = nbytes / elements_per_byte;
    // Cast input to PackType for aligned reads
    const auto* __restrict__ in_packed = reinterpret_cast<const PackType*>(in);

    // Process 16 output bytes at a time
    constexpr size_t unroll_factor = 16;
    const size_t main_loop         = out_bytes & ~(unroll_factor - 1);

#ifdef USE_OPENMP
#pragma omp parallel for if (Parallel)
#endif
    for (size_t i = 0; i < main_loop; i += unroll_factor) {
        // Prefetch next data
        __builtin_prefetch(&in_packed[i + unroll_factor], 0, 1);

#if defined(__clang__)
#pragma clang loop unroll(full) vectorize(enable)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#pragma GCC ivdep
#endif
        for (size_t j = 0; j < unroll_factor; ++j) {
            PackType x = in_packed[i + j];
            x          = (x & mask) * swar;
            out[i + j] = static_cast<uint8_t>(x >> shift);
        }
    }
    // Handle remaining bytes
    for (size_t i = main_loop; i < out_bytes; ++i) {
        PackType x = in_packed[i];
        x          = (x & mask) * swar;
        out[i]     = static_cast<uint8_t>(x >> shift);
    }
}

template <size_t NBits>
    requires(NBits == 1 || NBits == 2 || NBits == 4)
class BitPacker {
    template <BitOrder Order, bool Parallel, bool UseLookup>
    static void unpack_impl(std::span<const uint8_t> in,
                            std::span<uint8_t> out) {
        if constexpr (UseLookup) {
            unpack_lookup_impl<NBits, Order, Parallel>(in.data(), out.data(),
                                                       in.size());
        } else {
            unpack_compute_impl<NBits, Order, Parallel>(in.data(), out.data(),
                                                        in.size());
        }
    }

    template <BitOrder Order, bool Parallel, bool UseSWAR>
    static void pack_impl(std::span<const uint8_t> in, std::span<uint8_t> out) {
        if constexpr (UseSWAR) {
            pack_swar_impl<NBits, Order, Parallel>(in.data(), out.data(),
                                                   in.size());
        } else {
            pack_compute_impl<NBits, Order, Parallel>(in.data(), out.data(),
                                                      in.size());
        }
    }

public:
    static void unpack(std::span<const uint8_t> in,
                       std::span<uint8_t> out,
                       BitOrder bitorder = BitOrder::BigEndian,
                       bool parallel     = false,
                       bool use_lookup   = false) {
        const auto expected_size = in.size() * (8 / NBits);
        if (out.size() != expected_size) {
            throw std::invalid_argument(
                std::format("Output buffer size mismatch. Expected {}, got {}",
                            expected_size, out.size()));
        }

        // Dispatch based on runtime parameters
        if (bitorder == BitOrder::BigEndian) {
            if (parallel) {
                if (use_lookup) {
                    unpack_impl<BitOrder::BigEndian, true, true>(in, out);
                } else {
                    unpack_impl<BitOrder::BigEndian, true, false>(in, out);
                }
            } else {
                if (use_lookup) {
                    unpack_impl<BitOrder::BigEndian, false, true>(in, out);
                } else {
                    unpack_impl<BitOrder::BigEndian, false, false>(in, out);
                }
            }
        } else {
            if (parallel) {
                if (use_lookup) {
                    unpack_impl<BitOrder::LittleEndian, true, true>(in, out);
                } else {
                    unpack_impl<BitOrder::LittleEndian, true, false>(in, out);
                }
            } else {
                if (use_lookup) {
                    unpack_impl<BitOrder::LittleEndian, false, true>(in, out);
                } else {
                    unpack_impl<BitOrder::LittleEndian, false, false>(in, out);
                }
            }
        }
    }

    static void pack(std::span<const std::uint8_t> in,
                     std::span<std::uint8_t> out,
                     BitOrder bitorder = BitOrder::BigEndian,
                     bool parallel     = false,
                     bool use_swar     = false) {
        const auto expected_size = in.size() * NBits / 8;
        if (out.size() != expected_size) {
            throw std::invalid_argument(
                std::format("Output buffer size mismatch. Expected {}, got {}",
                            expected_size, out.size()));
        }

        // Dispatch based on runtime parameters
        if (bitorder == BitOrder::BigEndian) {
            if (parallel) {
                if (use_swar) {
                    pack_impl<BitOrder::BigEndian, true, true>(in, out);
                } else {
                    pack_impl<BitOrder::BigEndian, true, false>(in, out);
                }
            } else {
                if (use_swar) {
                    pack_impl<BitOrder::BigEndian, false, true>(in, out);
                } else {
                    pack_impl<BitOrder::BigEndian, false, false>(in, out);
                }
            }
        } else {
            if (parallel) {
                if (use_swar) {
                    pack_impl<BitOrder::LittleEndian, true, true>(in, out);
                } else {
                    pack_impl<BitOrder::LittleEndian, true, false>(in, out);
                }
            } else {
                if (use_swar) {
                    pack_impl<BitOrder::LittleEndian, false, true>(in, out);
                } else {
                    pack_impl<BitOrder::LittleEndian, false, false>(in, out);
                }
            }
        }
    };
};

/*
Function to unpack 1, 2 and 4 bit data into an 8-bit array.
*/
inline void unpack(std::span<const uint8_t> in,
                   std::span<uint8_t> out,
                   size_t nbits,
                   std::string_view bitorder = "big",
                   bool parallel             = false,
                   bool use_lookup           = false) {
    const auto order = parse_bit_order(bitorder);

    switch (nbits) {
    case 1:
        BitPacker<1>::unpack(in, out, order, parallel, use_lookup);
        break;
    case 2:
        BitPacker<2>::unpack(in, out, order, parallel, use_lookup);
        break;
    case 4:
        BitPacker<4>::unpack(in, out, order, parallel, use_lookup);
        break;
    default:
        throw std::invalid_argument(
            std::format("nbits must be 1, 2, or 4. Got {}", nbits));
    }
}

/*
Function to pack 1, 2 and 4 bit data into an 8-bit array.
*/
inline void pack(std::span<const uint8_t> in,
                 std::span<uint8_t> out,
                 size_t nbits,
                 std::string_view bitorder = "big",
                 bool parallel             = false,
                 bool use_swar             = false) {
    const auto order = parse_bit_order(bitorder);

    switch (nbits) {
    case 1:
        BitPacker<1>::pack(in, out, order, parallel, use_swar);
        break;
    case 2:
        BitPacker<2>::pack(in, out, order, parallel, use_swar);
        break;
    case 4:
        BitPacker<4>::pack(in, out, order, parallel, use_swar);
        break;
    default:
        throw std::invalid_argument(
            std::format("nbits must be 1, 2, or 4. Got {}", nbits));
    }
}

} // namespace numbits