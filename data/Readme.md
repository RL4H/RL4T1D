# Data Importing

This folder is set up to import and make use of the collected data from previously ran experiments.

Once you have downloaded the data, you will find it in a structure like the following:


```
--- simulated_data
    --- object_save
        --- data_dictionary_adolescent0_data.pkl
        --- data_dictionary_adolescent1_data.pkl
            ...
        --- data_dictionary_adult8_data.pkl
        --- data_dictionary_adult9_data.pkl

``` 

To run the sample scripts, alter the `DATA_DEST` variable at the top to the root of the data folder ('simulated_data' in the structure shown above). If you keep the files together as in the suggested download, the relative file reference should be correct.

`sample_script_1.py` performs a basic import on the data, going through each individual and printing the first trial preview as a raw numpy array.

`sample_script_2.py` counts the total time elpased for all trials.

`sample_script_3.py` converts the trials to a .csv format.

`sample_script_4.py` generates a table of mean and standard deviation for glucose and insulin data across individuals and algorithms.

To run these files, you will need the following modules:
- numpy
- pandas
- pickle
