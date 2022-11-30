#!/bin/bash

# This script is used to generate the wisdom file for FFTW3.

# have two nested for loops going from 10 to N
# where N is the largest dimensions of the FFTs you will be doing
for (( i=100; i<=110; i++ ))
do
   for (( j=100; j<=110; j++ ))
   do
        for k in o
        do
            # run the FFTW3 wisdom generator
            # and redirect the output to a file
            # this will take a while
            # fftw-wisdom -T 4 -v -o "wisdom" "c${i}f${j}x${k}"
            area=`expr $i \* $j`
            n="c${k}f${area}"
            # echo $n
            fftw-wisdom -T 4 -v -o "wisdom" "${n}"
        done
   done
done
