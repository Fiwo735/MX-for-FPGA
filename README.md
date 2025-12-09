# MX-for-FPGA
Implementation of Microscaling data formats in SystemVerilog.

## Simulation
From top-level directory:
```
$ ./tb/dot/<module_name>/run_sim.sh
```

## Synthesis
From top-level directory:
```
$ vivado -mode batch -source ./src/matmul/matmul_int/run_synth.tcl
```