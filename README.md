# ICOS
This repository is the code of ICOS.
In this repository, the code to repeat the experiments on MNIST, CIFAR10, and CIFAR100 is available.

The python code in MNISTCNNs.py, CIFAR10CNNs.py, and CIFAR100/CIFAR100CNNs.py can be executed to generate the .csv files containing the training, validation, and test sets ready to be submitted to ICOS. All the requirements needed to execute the Python code are listed in the requirements.txt file.

The implementation of ICOS provided needs an installation of KNIME (v4.0.0).
After the import of the .knar file into the KNIME workspace, the .csv file generated can be inserted in input in the "CSV Reader" blocks according to their tags.
With the "Double Configuration" blocks the minimum confidence and support can be set.
The "CSV Writer" blocks can be configured with the path where the CSV containing the output of the testing session can be saved.
The "Java Snippet" block, named "IDI", inside the "ICOS" metanode can be used to switch the partitioning criterion by uncommenting the corresponding code.
The "Partitioning" blocks, "Node 15" and "Node 77" (inside the "CRO" metanode), can be used to change the test set size for ICOS and CRO respectively. 

Due to Github repos space limitations, we uploaded only the M and N models for CIFAR100.
In case of acceptance, the other models and both the code and the images for ImageNet experiments will be uploaded via Google Drive.
