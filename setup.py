from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path)as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        hyphen= "-e ."
        if hyphen  in requirements:
            requirements.remove(hyphen)
setup(
     name='mlproject',
     version='0.0.1',
      author='adarsh',
      author_email='adarshkumar2605898@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')
      )