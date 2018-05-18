#!/bin/bash

checkpoint=$1
prefixes=('carry' 'ride' 'eat' 'sit' 'throw' 'park' 'lay' 'walk' 'hang' 'cover')

echo "---EVAL: FULL---"
echo "  sg:" 
cat ${checkpoint}/evaluate_full-sg.log | grep -E 'R@100|R@50|R@20'
echo "  pred:" 
cat ${checkpoint}/evaluate_full-pred.log | grep -E 'R@100|R@50|R@20'

for (( i=0; i < ${#prefixes[@]}; ++i)); do
    echo "---EVAL: ${prefixes[i]}---"
    echo "  sg:" 
    cat ${checkpoint}/evaluate_test-${prefixes[i]}-sg.log | grep -E 'R@100|R@50|R@20'
    echo "  pred:" 
    cat ${checkpoint}/evaluate_test-${prefixes[i]}-pred.log | grep -E 'R@100|R@50|R@20'
done
