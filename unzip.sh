mkdir -p data
# join forecast demand_nsw files into a single file.
cd data || exit
cat forecastdemand_nsw.csv.zip.part* > forecastdemand_nsw_csv.zip
# unzip the data files into the data folder
unzip data/forecastdemand_nsw_csv.zip -d data
unzip project/data/temperature_nsw_csv.zip -d data
unzip project/data/totaldemand_nsw_csv.zip -d data
