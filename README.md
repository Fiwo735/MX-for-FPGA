# MX-for-FPGA
Implementation of Microscaling data formats in SystemVerilog.

## How to run
### Simulation
```
MX-for-FPGA:~$ ./tb/dot/<module_name>/run_sim.sh
```

### Synthesis
```
MX-for-FPGA:~$ vivado -mode batch -source ./src/attention/attention_int/run_synth.tcl

On Kraken:
MX-for-FPGA:~$ nohup /mnt/applications/Xilinx/24.2/Vivado/2024.2/bin/vivado -mode batch -source ./src/attention/run_synth_fp.tcl -tclargs 2048 2048 32 32 32 8 4 3 4 3 4 3 NEUMAIER TWOSUM KLEIN yes yes yes > nohup_large.out 2>&1 &

```

### DSE
First time:
```
MX-for-FPGA:~$ python3 -m venv venv
MX-for-FPGA:~$ source venv/bin/activate
MX-for-FPGA:~$ pip install -r requirements.txt
MX-for-FPGA:~$ python DSE.py --verbose
```

Afterwards:
```
MX-for-FPGA:~$ source venv/bin/activate
MX-for-FPGA:~$ python DSE.py --verbose
```

For long running jobs:
```
MX-for-FPGA:~$ source venv/bin/activate
MX-for-FPGA:~$ nohup python -u DSE.py --verbose > DSE_run_$(date +%F_%H-%M-%S).log 2>&1 &
[1] XXXXXXX
```

Check XXXXXXX process status:
```
MX-for-FPGA:~$ ps -fp XXXXXXX
```