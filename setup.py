from setuptools import find_packages, setup
#will automatically find out all the packges that are available in entire ML application in the directory that we have created
from typing import List

#define get_requirements function 
def get_requirements(file_path:str)->List[str] : 
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj : 
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements : 
            requirements.remove("-e .")
    return requirements
    


#do the setup, with all the packages required
#whenever, we are creating a package, 
#we need to write what's the name of project/ application
#other parameters include version, description, long description
#can be considered as metadata information about the project
setup(
    name= "mlproject",
    version= "0.0.1",
    #we can keep on updating version
    author= "Sanjana",
    author_email= "sanjana.rawal2006@gmail.com",
    packages= find_packages(),
    #checks how many folders has __init__.py files in them, 
    #the folders with __init__.py file will be considered as a package & will try to build it
    #once, a folder is built as package, we can import it wherever we want
    #but, for doing so, we need to put it in PyPi package
    # install_requires = ["pandas", "numpy", "seaborn"] #suitable for few 
    # for a lot of packages , it can be installed like - 
    install_requires= get_requirements("requirements.txt")
)