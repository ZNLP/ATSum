#!/bin/bash

python mergebpe.py $1 $1.out
files2rouge $1.out reference > $2.rougescore
rm $1.out
