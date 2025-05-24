import numpy as np
import perfplot
import click

import numbits


@click.command()
@click.option("--test", default="unpack", help="Choose between 'unpack' and 'pack'")
@click.option("--bitorder", default="big", help="Choose between 'big' and 'little'")
@click.option(
    "--nbits",
    default=1,
    type=click.IntRange(min=1, max=4),
    help="Number of bits to pack/unpack",
)
def main(test="unpack", bitorder="big", nbits=1):
    if test == "unpack":
        kernels = [
            lambda arr: numbits.unpack(
                arr, nbits, bitorder=bitorder, parallel=False, use_lookup=False
            ),
            lambda arr: numbits.unpack(
                arr, nbits, bitorder=bitorder, parallel=True, use_lookup=False
            ),
            lambda arr: numbits.unpack(
                arr, nbits, bitorder=bitorder, parallel=False, use_lookup=True
            ),
            lambda arr: numbits.unpack(
                arr, nbits, bitorder=bitorder, parallel=True, use_lookup=True
            ),
        ]
        labels = [
            "numbits",
            "numbits_parallel",
            "numbits_lookup",
            "numbits_lookup_parallel",
        ]
        if nbits == 1:
            kernels.insert(0, lambda arr: np.unpackbits(arr, bitorder=bitorder))
            labels.insert(0, "numpy")
        bench_stat = perfplot.bench(
            setup=lambda n: np.random.randint(256, size=n, dtype="uint8"),
            n_range=[2**k for k in range(0, 24)],
            kernels=kernels,
            labels=labels,
            xlabel="n",
            title=f"Unpack {nbits} bit ({bitorder} endian)",
            target_time_per_measurement=1,
            equality_check=None,
        )
        bench_stat.save(
            f"benchmark_unpack_{nbits}bit_{bitorder}.png",
            transparent=False,
            bbox_inches="tight",
        )
    else:
        kernels = [
            lambda arr: numbits.pack(
                arr, nbits, bitorder=bitorder, parallel=False, use_swar=False
            ),
            lambda arr: numbits.pack(
                arr, nbits, bitorder=bitorder, parallel=True, use_swar=False
            ),
            lambda arr: numbits.pack(
                arr, nbits, bitorder=bitorder, parallel=False, use_swar=True
            ),
            lambda arr: numbits.pack(
                arr, nbits, bitorder=bitorder, parallel=True, use_swar=True
            ),
        ]
        labels = [
            "numbits",
            "numbits_parallel",
            "numbits_swar",
            "numbits_swar_parallel",
        ]
        if nbits == 1:
            kernels.insert(0, lambda arr: np.packbits(arr, bitorder=bitorder))
            labels.insert(0, "numpy")

        bench_stat = perfplot.bench(
            setup=lambda n: np.random.randint((1 << nbits) - 1, size=n, dtype="uint8"),
            n_range=[2**k for k in range(3, 24)],
            kernels=kernels,
            labels=labels,
            xlabel="n",
            title=f"Pack {nbits} bit ({bitorder} endian)",
            target_time_per_measurement=1,
            equality_check=None,
        )
        bench_stat.save(
            f"benchmark_pack_{nbits}bit_{bitorder}.png",
            transparent=False,
            bbox_inches="tight",
        )
    bench_stat.show()


if __name__ == "__main__":
    main()
