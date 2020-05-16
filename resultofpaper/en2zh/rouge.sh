#!/bin/bash

sed -i 's/ <\/t>//g' $1
sed -i 's/<t> //g' $1
python process.py $1
./chi_char_segment.pl -type plain < $1.new > $1.char
python transform_to_enchar.py $1.char reference
python rouge.py
files2rouge $1.char.en reference.en > $2.rougescore
rm $1.new $1.char $1.char.en reference.en
