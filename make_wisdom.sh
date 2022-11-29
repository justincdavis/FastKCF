#!/bin/bash

# This script is used to generate the wisdom file for FFTW3.

# have two nested for loops going from 10 to N
# where N is the largest dimensions of the FFTs you will be doing
for j in {10..250000}
do
#    for k in {10..1000}
#    do
        for i in o i
        do
            # run the FFTW3 wisdom generator
            # and redirect the output to a file
            # this will take a while
            # fftw-wisdom -T 4 -v -o "wisdom" "c${i}f${j}x${k}"
            fftw-wisdom -T 4 -v -w "wisdom" -o "wisdom" "c${i}f${j}"
        done
#    done
done
