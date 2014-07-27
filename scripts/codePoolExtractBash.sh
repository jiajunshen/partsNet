#!/bin/bash
for((i = 0;i<=7;i++))
do
    python poolExtractedFeature.py $i
done
