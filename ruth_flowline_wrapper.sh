#!/bin/bash
# wrapper for FlowlineMassCon.py

touch job.txt
rm -f job.txt
touch job.txt

ncore=6

mb=4
dhdt=-.5

for ela in {1400..1700..10}
do
   echo "python $FlowlineMassCon.py config.ini -mb $mb -ela $ela -dhdt $dhdt -out_name "mb_${mb}_ela_${ela}_dhdt_${dhdt}.csv"" >> job.txt
done

ela=1550
for dhdt in $(seq -2.5 .25 2.5)
do
   echo "python $FlowlineMassCon.py config.ini -mb $mb -ela $ela -dhdt $dhdt -out_name "mb_${mb}_ela_${ela}_dhdt_${dhdt}.csv"" >> job.txt
done

# parallel -j $ncore < ./job.txt

# rm job.txt