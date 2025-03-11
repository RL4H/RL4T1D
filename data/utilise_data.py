# from import_data import DataImporter

# all_data = DataImporter(verbose=True)

# for individual_data in all_data:
#     print("Imported data for", all_data.current_individual)
#     del individual_data


#EXAMPLE 1
# from import_data import DataImporter

# all_data = DataImporter(verbose=False)

# total_simulated_minutes = 0

# for individual_data in all_data:
#     print("Imported data for", all_data.current_individual)
#     individual_data.flatten()
#     raw_data = individual_data.get_raw()

#     individual_minutes = 0
#     for trial in raw_data:
#         rows, _ = trial.shape
#         individual_minutes += 5 * rows #each row is 5 minutes of data
    
#     print("Counted", individual_minutes, "minutes of data for", all_data.current_individual)
#     total_simulated_minutes += individual_minutes
#     del individual_data


# print(f"Total minutes: {total_simulated_minutes}m")
# years = total_simulated_minutes // (60 * 24 * 365)
# weeks = (total_simulated_minutes - (years * (60 * 24 * 365))) // (60 * 24 * 7)
# days = (total_simulated_minutes - (years * (60 * 24 * 365)) - (weeks * (60 * 24 * 7))) // (60 * 24)
# hours = (total_simulated_minutes % (60 * 24)) // 60
# minutes = (total_simulated_minutes % (60 * 24)) % 60
# print(f"Total time: {years}y {weeks} weeks, {days}d {hours}h {minutes}m")

#EXAMPLE 2
from import_data import DataImporter

all_data = DataImporter(verbose=True)

SAVE_PATH = "../data/raw_csv/"
SAVE_FLAT = True

for individual_data in all_data:
    if SAVE_FLAT:
        individual_data.flatten()
    individual_name = all_data.current_individual
    individual_data.save_as_csv(individual_name,SAVE_PATH)
    del individual_data

