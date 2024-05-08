from setuptools import setup, find_packages

def get_requirements_txt(requirements_path):
    with open(requirements_path, 'r') as file:
        requirements = [line.rstrip('\n') for line in file.readlines()]
    return requirements

setup( 
    name='mdetsyn', 
    version='0.0.4.post1', 
    description='Data Synthesis pipeline to generate object detection data', 
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    author='PD-Mera', 
    author_email='phuongdong1772000@gmail.com', 
    packages=find_packages(), 
    data_files=["requirements.txt"],
    include_package_data=True,
    install_requires=get_requirements_txt("requirements.txt")
) 