#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include "stdio.h"

int main()
{	
	k4a_playback_t playback_handle = NULL;
	if (k4a_playback_open("./Benchmark/sequences/woodenTable/woodenTable_10/woodenTable_10.mkv", &playback_handle) != K4A_RESULT_SUCCEEDED)  // The path of mkv files
	{
    	printf("Failed to open recording\n");
    	return 1;
	}
	k4a_imu_sample_t imu_sample;
	k4a_stream_result_t result = K4A_STREAM_RESULT_SUCCEEDED;
	
	
	FILE *fpWrite=fopen("./Benchmark/sequences/woodenTable/woodenTable_10/data.txt","w");  // The save path of imu data
	if(fpWrite==NULL)  
	{  
	   return 0;  
	} 


	int step = 0;
	
	while (result == K4A_STREAM_RESULT_SUCCEEDED)
	{
	    result = k4a_playback_get_next_imu_sample(playback_handle, &imu_sample);
	    if (result == K4A_STREAM_RESULT_SUCCEEDED)
	    {

		fprintf(fpWrite,"%f,%f,%f,%f,%f,%f,%f\n",0.000001*imu_sample.acc_timestamp_usec,imu_sample.acc_sample.xyz.x,imu_sample.acc_sample.xyz.y,imu_sample.acc_sample.xyz.z,imu_sample.gyro_sample.xyz.x,imu_sample.gyro_sample.xyz.y,imu_sample.gyro_sample.xyz.z);  

	
	    }
	    else if (result == K4A_STREAM_RESULT_EOF)
	    {
	       fclose(fpWrite);

		break;
	    }
	    step+=1;
	}

	if (result == K4A_STREAM_RESULT_FAILED)
	{
	    printf("Failed to read entire recording\n");
	    return 1;
	}


}
