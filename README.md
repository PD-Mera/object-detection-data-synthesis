# Object Detection Data Synthesis

## How to run

### Run with pip

``` bash
pip install mdetsyn
```

And run in python file 

``` python
from mdetsyn import run_synthesis

run_synthesis(
    backgrounds_dir="./backgrounds", 
    objects_dir="./objects, 
    synthetic_save_dir="./synthesis", 
    synthetic_number=1000, 
    # class_mapping_path="./class_mapping.json", 
    # class_txt_path="./classes.txt"
)
```

### Run with command line

``` bash
python synthesis.py --backgrounds ./backgrounds --objects ./objects --savename ./synthesis --class_mapping ./class_mapping.json --number 1000
```