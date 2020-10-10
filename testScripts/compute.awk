{
	printf("%f %f -- ", $1, $2);
	total = 0
	for(i=5; i<=NF; i++) { total = total + $i; }
	printf("%d -- ", total);
	i = 5;
	for(r=0; r<16; r++)
	{
		total = 0
		for(i=0; i<8; i++)
		{
			idx = 5 + r*8 + i
			total = total + $idx;
		}
		printf("%d ", total/8);
	}
	printf("\n");
}
