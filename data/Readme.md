# Data Importing

This folder is set up to import and make use of the collected data from previously ran experiments.

The data itself is too large to be stored in the repository, so to use it on your local machine you will have to download the files from onedrive and put them in the correct place. The file structure is as follows:

```
--- data
    --- adolescent
        --- A2C
        --- AUXML
        --- BBHE
        --- BBI
        --- G2P2C
        --- PPO
        --- SAC
    --- adult
        --- A2C
        --- AUXML
        --- BBHE
        --- BBI
        --- G2P2C
        --- PPO
        --- SAC
``` 

Additionally, if the SAVE_TO_PICKLE variable is set to true when importing the data (running `import_data.py` as main), a pickled version of the data will be stored in `object_save/object_dictionary.pkl`.

The `import_data.py` file is set up to be ran from the experiments folder in following with the other files in the repository, so keep this in mind when running. ie.

```
>>> cd experiments
>>> python ../data/import_data.py
```

To run this file, you will need the pickle module on top of the other modules needed for this repository.

In the main execution of the `import_data.py` file, there are examples of how to read and write the data.