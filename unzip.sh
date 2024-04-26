cd ../data/NSW || exit
cat forecastdemand_nsw.csv.zip.part* > forecastdemand_nsw_csv.zip
unzip -o forecastdemand_nsw_csv.zip
unzip -o temperature_nsw_csv.zip
unzip -o totaldemand_nsw_csv.zip
