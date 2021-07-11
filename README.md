# VOLK Pothos Blocks

This toolkit adds Pothos blocks that wrap the majority of functionality
from the [Vector-Optimized Library of Kernels (VOLK)](https://www.libvolk.org/).
There is significant overlap with [PothosComms](https://github.com/pothosware/PothosComms),
but this module optimizes for various cases not covered by PothosComms, such
as operations taking in ports of different types.

Depending on the block and underlying architecture, these blocks' performance
may be comparable to their PothosComms equivalent's SIMD implementations or
vary significantly.

## Dependencies

* Pothos library (0.7+)
* VOLK

## Licensing information

This module is licensed under the GNU General Public License v3.0. To view the
full license, view the `COPYING` file.
