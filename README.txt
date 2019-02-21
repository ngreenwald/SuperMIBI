# Erin and Noah's super great project
# To get started locally, git yer github synced up
# Then execute the following command from terminal (assuming you have anaconda installed):  
	$conda env create -f environment.yml
# Create a new pycharm project, but click use existing interpreter.
# Select home/anaconda/envs/SuperMIBIConda/bin/python as your interpreter


# For AWS:
# Need to decide best way to track separate users
# The MacOSX virtual environment won't work on conda. 
# To generate a new virtual environment template:
# $ conda create -n SuperMIBIAWS numpy tensorflow matplotlib=2.2.2 scikit-image pandas nomkl
# $ conda activate SuperMIBIAWS
# $ conda env export > AWS_Environment.yml
# 
# to use the existing file:
# $ conda env create -f AWS_Environment.yml
# $ conda activate SuperMIBIAWS
# $ conda deactivate SuperMIBIAWS
# However, somtimes this doesn't work? Instead use:
# $ source activate SuperMIBIAWS

# syntax:
	To copy files to/from AWS: $ scp -i <keypair> file_name_local.txt ubuntu@ec2-bla-bla.com~/folder_name


# to create a custom conda env with specified files:

