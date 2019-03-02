#!/bin/bash

#set -x

for var in "$@"; do

	cat $var | while read line; do
		url=$( echo $line | cut -d' ' -f1)
		filename=$( echo $line | cut -d' ' -f2)    
		ext="${filename##*.}"
		#echo $url, $filename
		
		wget -q --show-progress $url -O $filename
		
		echo "Unpacking $filename"
		if [ "$ext" == "zip" ]; then
			unzip -q -o $filename    
		fi    
		if [ "$ext" == "tgz" ]; then
			tar xzf $filename    
		fi
		
		echo "Removing  $filename"
		rm $filename    
	done

done
