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
        --- TD3
        --- DPG
        --- DDPG
    --- adult
        --- A2C
        --- AUXML
        --- BBHE
        --- BBI
        --- G2P2C
        --- PPO
        --- SAC
        --- TD3
``` 

Add a variable in the .env file called `SIM_DATA_PATH`, and set it to the root of the data folder stored on your local machine.
The BBHE and BBI folders contain csv files in the format `logs_worker_[individual number]_[trial number].csv`.

All other folders contain sub folders following the format `[model name][individual number]_[seed number].csv`.

If you are setting up the data for yourself, take care to remove other subfolders between the model folders and the files/folders specified above.

Additionally, if the SAVE_TO_PICKLE variable is set to true when importing the data (running `import_data.py` as main), a pickled version of the data will be stored in `object_save/object_dictionary.pkl`.

To run this file, simply navigate to the root of the repo and run the following command:

```
>>> python /data/import_data.py
```

To run this file, you will need the pickle module on top of the other modules needed for the rest of this repository.

In the `utilise_data.py` file, there are examples of how to read and write the data.