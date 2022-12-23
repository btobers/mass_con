#!/bin/bash
# wrapper for FlowlineMassCon.py

touch job.txt
rm -f job.txt
touch job.txt

ncore=20
# plot="-plot"
plot=""
out_path="/zippy/MARS/targ/modl/mass_con/ruth/out/"
# out_path="C:/Users/btober/OneDrive/Documents/MARS/targ/modl/mass_con/ruth/out/"

# start time
start=$SECONDS

# beautiful nested nested nested nested for looooooop
counter=0
for ela in {1450..1650..25}
do
   for mb in {5..15..1}
   do
      for dhdt in $(seq -1.00 .25 0.00)
      do
         for gamma in $(seq 0.90 0.05 1.00)
         do
            # update counter
            counter=$[$counter + 1]
            echo "python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "${out_path}mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"" >> job.txt
         done
      done
   done
done

mb=10
dhdt=-0.50
gamma=0.90
# now do each variable independently - first ela location
for ela in {1450..1650..10}
do
   counter=$[$counter + 1]
   echo "python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "${out_path}ela/mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"" >> job.txt
done

ela=1550
# mb gradient
for mb in {5..15..1}
do
   counter=$[$counter + 1]
   echo "python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "${out_path}mb/mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"" >> job.txt
done

mb=10
# dhdt
for dhdt in $(seq -1.00 .25 0.00)
do
   counter=$[$counter + 1]
   echo "python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "${out_path}dhdt/mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"" >> job.txt
done

dhdt=-0.50
for gamma in $(seq 0.90 0.05 1.00)
do
   counter=$[$counter + 1]
   echo "python FlowlineMassCon.py config.ini $plot -mb $mb -ela $ela -dhdt $dhdt -gamma $gamma -out_name "${out_path}gamma/mb_${mb}_ela_${ela}_dhdt_${dhdt}_gamma_$gamma.csv"" >> job.txt
done

parallel -j $ncore < ./job.txt

# completion time
if (( $SECONDS > 3600 )) ; then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed $counter runs in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $SECONDS > 60 )) ; then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed $counter runs in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed $counter runs in $SECONDS seconds"
fi

# rm job.txt