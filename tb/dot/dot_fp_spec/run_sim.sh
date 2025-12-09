#!/bin/bash

dir_name="dot"
module_name="dot_fp_spec"
dir_module_name=$dir_name"/"$module_name
out_dir_name="tb/"$dir_module_name"/output"
src_file="src/"$dir_module_name".sv"
tb_file="tb/"$dir_module_name"/"$module_name"_tb.sv"
# util_package="tb/util/common_pkg.sv"

mkdir -p "$out_dir_name" && cd "$out_dir_name"

xvlog -sv ../../../../"$src_file" ../../../../"$tb_file"
xelab -sv "$module_name"_tb -R 