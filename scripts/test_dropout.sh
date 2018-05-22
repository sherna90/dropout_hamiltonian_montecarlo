#!/bin/bash
declare -a train="../data/adience.csv"
declare -a test="../data/adience_t.csv"
declare -a iterations="1000 10000"
declare -a lambda="100.0 10.0 1.0 0.1"
declare -a w="100"
declare -a sz="0.01"
declare -a ns="100"
declare -a n="true" 
declare -a s="true"
declare -a np="50"
declare -a mask_rate="0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 00.85 0.9 0.95"
declare -a repeat="1 2 3"

echo $train 
for i in $iterations
do
	for l in $lambda
	do
		for mr in $mask_rate
		do
			for r in $repeat
			do
				echo $r
				../build/test_all_dropout -train $train -test $test -l $l -w $w -nit $i -sz $sz -ns $ns -np $np -n $n -s $s -mr $mr
			done
		done
	done
done
