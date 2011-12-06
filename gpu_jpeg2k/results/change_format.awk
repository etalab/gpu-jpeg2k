#!/usr/bin/awk -f

BEGIN {
}
{
	if($1 ~ /ppm$/)	{
				print $1;
			}
	if($1 ~ /^[0-9]+/) { 
				sec=substr($1, 3, 2);
				mili=substr($1, 6);
				print sec*1000+mili*10;
			}
}
END {
}
