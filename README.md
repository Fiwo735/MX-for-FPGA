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
```

### DSE
First time:
```
MX-for-FPGA:~$ python3 -m venv venv
MX-for-FPGA:~$ source venv/bin/activate
MX-for-FPGA:~$ pip install -r requirements.txt
MX-for-FPGA:~$ python DSE.py
```

Afterwards:
```
MX-for-FPGA:~$ source venv/bin/activate
MX-for-FPGA:~$ python DSE.py
```

For long running jobs:
```
MX-for-FPGA:~$ source venv/bin/activate
MX-for-FPGA:~$ nohup python -u DSE.py > DSE_run_$(date +%F_%H-%M-%S).log 2>&1 &
[1] XXXXXXX
```

Check XXXXXXX process status:
```
MX-for-FPGA:~$ ps -fp XXXXXXX
```

1741603