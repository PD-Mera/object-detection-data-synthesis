# Object Detection Data Synthesis

Data Synthesis pipeline to generate object detection data

<a href="https://pypi.org/project/mdetsyn/"><img alt="Alt text" src="https://img.shields.io/badge/PyPI-3775A9.svg?style=for-the-badge&logo=PyPI&logoColor=white"/></a>

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