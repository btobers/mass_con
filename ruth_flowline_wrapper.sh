#!/bin/bash
# wrapper for FlowlineMassCon.py

touch job.txt
rm -f job.txt
touch job.txt

ncore=6
plot="-plot"
plot=""
mb=10
dhdt=-0.5
gamma=0.9

# loop over ela locations
for ela in {1400..1700..10}
do
   python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"
done

# loop over gamma
ela=1550
for gamma in $(seq 0.8 0.05 1.0)
do
   python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"
done

# loop over dhdt
gamma=0.9
for dhdt in $(seq -1.5 .25 1.5)
do
   python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"
done

# parallel -j $ncore < ./job.txt

# rm job.txt