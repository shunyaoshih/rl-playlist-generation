# !/bin/bash
cd data
python3 data_utils.py 30 70000
python3 tf_format.py 30
