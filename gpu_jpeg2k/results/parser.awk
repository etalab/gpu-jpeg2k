#!/usr/bin/awk -f

BEGIN {
sum = 0.0;
lines = 0;
}
{
	if($1 ~ /ppm/) {
				if(lines != 0)
					print sum/lines;
				#if(match($1, "//"))
					printf("%s ", substr($1, RSTART));
				sum = 0.0;
				lines = 0;
			}
	if($1 ~ /^[0-9]+/) { 
				sum += $1;
				lines++;
			}
}
END {
	print sum/lines;
}
