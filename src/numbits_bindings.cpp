#include <cstdint>
#include <span>
#include <string_view>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "numbits/numbits.hpp"

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

PYBIND11_MODULE(numbits, m) {
    m.doc() = "Pack and unpack 1, 2 and 4 bit data into/from an 8-bit array.";

    m.def(
        "unpack_buffered",
        [](const py::array_t<uint8_t, py::array::c_style>& in,
           py::array_t<uint8_t, py::array::c_style>& out, size_t nbits,
           std::string_view bitorder = "big", bool parallel = false,
           bool use_lookup = false) {
            numbits::unpack(std::span<const uint8_t>(in.data(), in.size()),
                            std::span<uint8_t>(out.mutable_data(), out.size()),
                            nbits, bitorder, parallel, use_lookup);
        },
        "in"_a, "out"_a, "nbits"_a, "bitorder"_a = "big", "parallel"_a = false,
        "use_lookup"_a = false,
        "Unpack 1, 2 and 4-bit data from an 8-bit numpy array into a "
        "pre-allocated buffer. If use_lookup is true, uses a lookup table to "
        "unpack the data.");

    m.def(
        "unpack",
        [](const py::array_t<uint8_t, py::array::c_style>& in, size_t nbits,
           std::string_view bitorder = "big", bool parallel = false,
           bool use_lookup = false) {
            auto outarray = py::array_t<uint8_t, py::array::c_style>(
                static_cast<ssize_t>(in.size() * 8 / nbits));
            numbits::unpack(
                std::span<const uint8_t>(in.data(), in.size()),
                std::span<uint8_t>(outarray.mutable_data(), outarray.size()),
                nbits, bitorder, parallel, use_lookup);
            return outarray;
        },
        "in"_a, "nbits"_a, "bitorder"_a = "big", "parallel"_a = false,
        "use_lookup"_a = false,
        "Unpack 1, 2 and 4-bit data from an 8-bit numpy array. If use_lookup "
        "is true, uses a lookup table to unpack the data.");

    m.def(
        "pack_buffered",
        [](const py::array_t<uint8_t, py::array::c_style>& in,
           py::array_t<uint8_t, py::array::c_style>& out, size_t nbits,
           std::string_view bitorder = "big", bool parallel = false,
           bool use_swar = false) {
            numbits::pack(std::span<const uint8_t>(in.data(), in.size()),
                          std::span<uint8_t>(out.mutable_data(), out.size()),
                          nbits, bitorder, parallel, use_swar);
        },
        "in"_a, "out"_a, "nbits"_a, "bitorder"_a = "big", "parallel"_a = false,
        "use_swar"_a = false,
        "Pack 1, 2 and 4-bit data into an pre-allocated 8-bit numpy array. \n"
        "If use_swar is true, uses a SWAR algorithm to pack the data.");

    m.def(
        "pack",
        [](const py::array_t<uint8_t, py::array::c_style>& in, size_t nbits,
           std::string_view bitorder = "big", bool parallel = false,
           bool use_swar = false) {
            auto outarray = py::array_t<uint8_t, py::array::c_style>(
                static_cast<ssize_t>(in.size() * nbits / 8));
            numbits::pack(
                std::span<const uint8_t>(in.data(), in.size()),
                std::span<uint8_t>(outarray.mutable_data(), outarray.size()),
                nbits, bitorder, parallel, use_swar);
            return outarray;
        },
        "in"_a, "nbits"_a, "bitorder"_a = "big", "parallel"_a = false,
        "use_swar"_a = false,
        "Pack 1, 2 and 4-bit data into an 8-bit numpy array. \n"
        "If use_swar is true, uses a SWAR algorithm to pack the data.");
}
