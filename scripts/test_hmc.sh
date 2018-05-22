#!/bin/bash
declare -a train="../data/adience.csv"
declare -a test="../data/adience_t.csv"
declare -a iterations="1000"
declare -a lambda="100.0 10.0 1.0 0.1 0.01"
declare -a w="100"
declare -a stepsize="0.005 0.001 0.0005 0.0001 0.00001"
declare -a numsteps="100 50 10"
declare -a n="true" 
declare -a s="true"
declare -a np="30"
declare -a repeat="1"

echo $train 
for i in $iterations
do
	for l in $lambda
	do
		for sz in $stepsize
		do
			for ns in $numsteps
			do
				for r in $repeat
				do
					echo $r
					../build/test_all_hmc -train $train -test $test -l $l -w $w -nit $i -sz $sz -ns $ns -np $np -n $n -s $s
				done
			done
		done
	done
done
