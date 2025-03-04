from import_data import DataImporter

all_data = DataImporter(verbose=True)

for data in all_data:
    print("Imported data for", all_data.current_individual)
