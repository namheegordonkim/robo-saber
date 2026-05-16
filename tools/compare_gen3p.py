"""Compare two gen3p.nc files group-by-group with tolerance.

Usage: python tools/compare_gen3p.py <gold_path> <new_path>

Each iteration of robo-saber/generate.py writes one NetCDF group keyed by the
output index, so we walk every top-level group, then every variable / attr in
that group. Floats use np.allclose(rtol, atol); ints / strings use exact
equality. Per-variable max |delta| is printed for floats. Exits non-zero on
the first out-of-tolerance mismatch.
"""

import sys

import h5py
import numpy as np
import xarray as xr

RTOL = 1e-4
ATOL = 1e-5


def list_groups(path):
    with h5py.File(path, "r") as f:
        return sorted(f.keys(), key=lambda k: (len(k), k))


def open_group(path, group):
    return xr.open_dataset(path, group=group, engine="h5netcdf")


def compare_group(group, ds_a, ds_b):
    fail = False

    var_names = sorted(set(ds_a.data_vars) | set(ds_b.data_vars))
    for name in var_names:
        if name not in ds_a or name not in ds_b:
            print(f"  [{group}] var {name!r} missing in one side")
            fail = True
            continue
        a = np.asarray(ds_a[name].values)
        b = np.asarray(ds_b[name].values)
        if a.shape != b.shape:
            print(f"  [{group}] var {name!r} shape mismatch: {a.shape} vs {b.shape}")
            fail = True
            continue
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            max_abs = float(diff.max()) if diff.size else 0.0
            ok = np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
            tag = "ok " if ok else "BAD"
            print(f"  [{group}] {tag} {name!r:24s} max|d|={max_abs:.3e} shape={a.shape}")
            if not ok:
                fail = True
        else:
            ok = np.array_equal(a, b)
            tag = "ok " if ok else "BAD"
            print(f"  [{group}] {tag} {name!r:24s} (exact, dtype={a.dtype})")
            if not ok:
                fail = True

    attr_names = sorted(set(ds_a.attrs) | set(ds_b.attrs))
    for name in attr_names:
        va = ds_a.attrs.get(name)
        vb = ds_b.attrs.get(name)
        if va != vb:
            print(f"  [{group}] BAD attr {name!r}: {va!r} vs {vb!r}")
            fail = True

    return fail


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <gold.nc> <new.nc>", file=sys.stderr)
        sys.exit(2)
    gold_path, new_path = sys.argv[1], sys.argv[2]

    gold_groups = list_groups(gold_path)
    new_groups = list_groups(new_path)
    print(f"gold groups: {gold_groups}")
    print(f"new  groups: {new_groups}")
    if gold_groups != new_groups:
        print("ERROR: group sets differ")
        sys.exit(1)

    fail = False
    for group in gold_groups:
        print(f"=== group {group} ===")
        with open_group(gold_path, group) as ds_a, open_group(new_path, group) as ds_b:
            if compare_group(group, ds_a, ds_b):
                fail = True

    if fail:
        print(f"\nFAIL: out-of-tolerance mismatch (rtol={RTOL}, atol={ATOL})")
        sys.exit(1)
    print(f"\nOK: all groups within tolerance (rtol={RTOL}, atol={ATOL})")


if __name__ == "__main__":
    main()
