# Script to generate multiple testcases - max value can correspondingly be changed.

max=100
for i in `seq 1 $max`
do
    ./a.out > testbed/$i
    echo "$i"
done
