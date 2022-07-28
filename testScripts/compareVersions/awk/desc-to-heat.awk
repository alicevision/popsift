BEGIN {
	x = -1.5;
	y = -1.5;
}

{
	if( NF == 10 )
	{
		printf("%.1f %.1f %f\n", x, y, $COL);
		x = x + 1;
		if( x > 2 )
		{
			printf("\n");
			x = -1.5;
			y = y + 1;
		}
	}
}

END {
	printf("\n");
}

