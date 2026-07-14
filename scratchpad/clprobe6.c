// QVAC-21623 Phase 3: does cl_qcom_recordable_queues survive the Adreno matmul pattern?
// The adreno mul_mat creates a cl image OVER a buffer, enqueues a kernel reading it, then
// RELEASES the image per call. Test: record such a kernel, release the image, replay -> OK or broken?
// Also test: update the underlying buffer contents between replays (must flow through).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#define CL_QUEUE_RECORDABLE_QCOM (1u << 30u)
typedef struct _cl_recording_qcom * cl_recording_qcom;
typedef struct { cl_uint dispatch_index; cl_uint arg_index; size_t arg_size; const void *arg_value; } cl_array_arg_qcom;
typedef struct { cl_uint dispatch_index; const size_t *p; } cl_offset_qcom;
typedef struct { cl_uint dispatch_index; const size_t *workgroup_size; } cl_workgroup_qcom;
typedef cl_recording_qcom (CL_API_CALL *pfnNew)(cl_command_queue, cl_int*);
typedef cl_int (CL_API_CALL *pfnEnd)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *pfnRel)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *pfnEnq)(cl_command_queue, cl_recording_qcom, size_t, const cl_array_arg_qcom*,
        size_t, const cl_offset_qcom*, size_t, const cl_workgroup_qcom*, size_t, const cl_workgroup_qcom*,
        cl_uint, const cl_event*, cl_event*);

int main(void){
    cl_platform_id plat; cl_device_id dev; cl_int err;
    clGetPlatformIDs(1,&plat,NULL); clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,NULL);
    void *dl = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    pfnNew pNew=(pfnNew)dlsym(dl,"clNewRecordingQCOM");
    pfnEnd pEnd=(pfnEnd)dlsym(dl,"clEndRecordingQCOM");
    pfnRel pRel=(pfnRel)dlsym(dl,"clReleaseRecordingQCOM");
    pfnEnq pEnq=(pfnEnq)dlsym(dl,"clEnqueueRecordingQCOM");
    cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,NULL,&err);
    cl_queue_properties rp[]={CL_QUEUE_PROPERTIES,CL_QUEUE_RECORDABLE_QCOM,0};
    cl_command_queue rq=clCreateCommandQueueWithProperties(ctx,dev,rp,&err);

    // kernel reads a float4 image1d_buffer, writes element-sum+1 to out buffer
    const char*src="__kernel void rd(__read_only image1d_buffer_t im,__global float*o){"
                   "int i=get_global_id(0); float4 v=read_imagef(im,i); o[i]=v.x+v.y+v.z+v.w;}";
    cl_program pr=clCreateProgramWithSource(ctx,1,&src,NULL,&err);
    if(clBuildProgram(pr,1,&dev,NULL,NULL,NULL)!=CL_SUCCESS){printf("build fail\n");return 1;}
    cl_kernel k=clCreateKernel(pr,"rd",&err);

    // buffer of 8 float4 = 32 floats; each pixel sums to 4*base
    float A[32]; for(int i=0;i<32;i++) A[i]=(float)(i/4); // pixel p -> {p,p,p,p} sum=4p... adjust
    for(int p=0;p<8;p++) for(int c=0;c<4;c++) A[p*4+c]=(float)p; // pixel p all = p, sum=4p
    float out[8]={0};
    cl_mem bufA=clCreateBuffer(ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(A),A,&err);
    cl_mem bufO=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,sizeof(out),NULL,&err);

    cl_image_format fmt={CL_RGBA,CL_FLOAT};
    cl_image_desc d; memset(&d,0,sizeof(d)); d.image_type=CL_MEM_OBJECT_IMAGE1D_BUFFER; d.image_width=8; d.buffer=bufA;
    cl_mem imgA=clCreateImage(ctx,CL_MEM_READ_ONLY,&fmt,&d,NULL,&err);
    printf("create image over buffer err=%d\n",err);

    clSetKernelArg(k,0,sizeof(cl_mem),&imgA);
    clSetKernelArg(k,1,sizeof(cl_mem),&bufO);
    size_t gws=8;
    cl_recording_qcom rec=pNew(rq,&err);
    err=clEnqueueNDRangeKernel(rq,k,1,NULL,&gws,NULL,0,NULL,NULL);
    printf("record enqueue(image kernel) err=%d\n",err);
    err=pEnd(rec); printf("endRecording err=%d\n",err);

    // *** THE TEST: release the image (as the adreno path does) BEFORE replay ***
    clReleaseMemObject(imgA);
    printf("released image; now replaying...\n");
    err=pEnq(q,rec,0,NULL,0,NULL,0,NULL,0,NULL,0,NULL,NULL);
    err|=clFinish(q);
    printf("replay-after-release enqueue/finish err=%d\n",err);
    clEnqueueReadBuffer(q,bufO,CL_TRUE,0,sizeof(out),out,0,NULL,NULL);
    int ok1=(out[0]==0.0f && out[7]==28.0f); // pixel p sum=4p -> out[7]=28
    printf("out={%.0f,%.0f,..,%.0f} expect{0,4,..,28} -> %s\n",out[0],out[1],out[7],ok1?"OK (release survived!)":"WRONG/BROKEN");

    // update buffer contents (in place) and replay again -> must reflect new data if image aliases buffer
    for(int p=0;p<8;p++) for(int c=0;c<4;c++) A[p*4+c]=(float)(p+100);
    clEnqueueWriteBuffer(q,bufA,CL_TRUE,0,sizeof(A),A,0,NULL,NULL);
    memset(out,0,sizeof(out));
    err=pEnq(q,rec,0,NULL,0,NULL,0,NULL,0,NULL,0,NULL,NULL); clFinish(q);
    clEnqueueReadBuffer(q,bufO,CL_TRUE,0,sizeof(out),out,0,NULL,NULL);
    int ok2=(out[7]==4.0f*107.0f);
    printf("after buffer update out[7]=%.0f expect %.0f -> %s\n",out[7],4.0f*107.0f,ok2?"OK (content flows)":"stale/WRONG");

    printf("VERDICT: image-release-then-replay = %s ; content-update = %s\n",
           ok1?"SURVIVES":"BREAKS", ok2?"FLOWS":"STALE");
    pRel(rec);
    return 0;
}
