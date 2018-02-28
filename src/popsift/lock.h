struct Lock
{
    int *mutex;
    
    Lock( void )
    {
	int state = 0;
	cudaMalloc((void**)& mutex,sizeof(int));
	cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    ~Lock( void )
    {
	cudaFree( mutex );
    }
    
    __device__ void
    lock( void )
    {
	while( atomicCAS( mutex, 0, 1 ) != 0 );
    }
    
    __device__ void
    unlock( void)
    {
        //atomicExch( mutex, 1 ); //crashes the kernel with error code:
	//CUDA error: the launch timed out and was terminated
	if ( *mutex == 1)
	    *mutex = 0;
    } 
};
