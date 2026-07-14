// QVAC-21623 Phase 2: functional probe for cl_qcom_recordable_queues on Adreno 740.
// Record a 2-dispatch sequence, replay, then replay WITH an arg update (the capability the
// decoder needs for changing per-token inputs). Verify output each time. R14: prove functional.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

// Try 3 resolution methods (Qualcomm proprietary fns often only via the deprecated
// clGetExtensionFunctionAddress or direct dlsym on the vendor .so):
static void *g_dl = NULL;
static void *resolve(cl_platform_id plat, const char *nm) {
    void *p = clGetExtensionFunctionAddressForPlatform(plat, nm);
    if (p) { printf("  %s via ForPlatform\n", nm); return p; }
    p = clGetExtensionFunctionAddress(nm);
    if (p) { printf("  %s via clGetExtensionFunctionAddress\n", nm); return p; }
    if (g_dl) { p = dlsym(g_dl, nm); if (p) { printf("  %s via dlsym\n", nm); return p; } }
    printf("  %s UNRESOLVED (all 3 methods)\n", nm);
    return NULL;
}

// --- cl_qcom_recordable_queues (not in Khronos headers; from Adreno SDK / MNN) ---
#define CL_QUEUE_RECORDABLE_QCOM            (1u << 30u)
#define CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE 0x41DE
typedef struct _cl_recording_qcom * cl_recording_qcom;
typedef struct { cl_uint dispatch_index; cl_uint arg_index; size_t arg_size; const void *arg_value; } cl_array_arg_qcom;
typedef struct { cl_uint dispatch_index; const size_t *global_work_offset; } cl_offset_qcom;
typedef struct { cl_uint dispatch_index; const size_t *workgroup_size; } cl_workgroup_qcom;
typedef cl_recording_qcom (CL_API_CALL *pfnNewRec)(cl_command_queue, cl_int*);
typedef cl_int (CL_API_CALL *pfnEndRec)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *pfnRelRec)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *pfnEnqRec)(cl_command_queue, cl_recording_qcom,
        size_t, const cl_array_arg_qcom*, size_t, const cl_offset_qcom*,
        size_t, const cl_workgroup_qcom*, size_t, const cl_workgroup_qcom*,
        cl_uint, const cl_event*, cl_event*);

int main(void) {
    cl_platform_id plat; cl_device_id dev; cl_int err;
    clGetPlatformIDs(1, &plat, NULL);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    char name[256]={0}; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("device: %s\n", name);

    cl_uint maxrec = 0;
    err = clGetDeviceInfo(dev, CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE, sizeof(maxrec), &maxrec, NULL);
    printf("CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE = %u (err=%d)  [decoder needs ~197/token]\n", maxrec, err);

    g_dl = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    printf("dlopen vendor libOpenCL.so = %p\n", g_dl);
    pfnNewRec pNew = (pfnNewRec)resolve(plat, "clNewRecordingQCOM");
    pfnEndRec pEnd = (pfnEndRec)resolve(plat, "clEndRecordingQCOM");
    pfnRelRec pRel = (pfnRelRec)resolve(plat, "clReleaseRecordingQCOM");
    pfnEnqRec pEnq = (pfnEnqRec)resolve(plat, "clEnqueueRecordingQCOM");
    printf("RESOLVED New=%p End=%p Release=%p Enqueue=%p\n", (void*)pNew,(void*)pEnd,(void*)pRel,(void*)pEnq);
    if (!pNew||!pEnd||!pRel||!pEnq) { printf("VERDICT: UNSUPPORTED (entry points unresolved)\n"); return 0; }

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    cl_queue_properties recprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_RECORDABLE_QCOM, 0 };
    cl_command_queue rq = clCreateCommandQueueWithProperties(ctx, dev, recprops, &err);
    printf("create recordable queue err=%d\n", err);
    if (err != CL_SUCCESS) { printf("VERDICT: UNSUPPORTED (recordable queue create failed)\n"); return 0; }

    const char *src =
      "__kernel void inc(__global float*a,__global float*b){int i=get_global_id(0);b[i]=a[i]+1.0f;}\n"
      "__kernel void mul2(__global float*a,__global float*b){int i=get_global_id(0);b[i]=a[i]*2.0f;}\n";
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (clBuildProgram(prog, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) { printf("build fail\n"); return 1; }
    cl_kernel kInc = clCreateKernel(prog, "inc", &err);
    cl_kernel kMul = clCreateKernel(prog, "mul2", &err);

    float A[8]={0,1,2,3,4,5,6,7}, A2[8]={10,11,12,13,14,15,16,17}, out[8]={0};
    cl_mem bA  = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(A),  A,  &err);
    cl_mem bA2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(A2), A2, &err);
    cl_mem bT  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(A), NULL, &err);  // inc: A->T
    cl_mem bO  = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(A), NULL, &err);  // mul2: T->O
    // dispatch 0: inc(bA -> bT) ; dispatch 1: mul2(bT -> bO)  => O = (A+1)*2
    clSetKernelArg(kInc,0,sizeof(cl_mem),&bA); clSetKernelArg(kInc,1,sizeof(cl_mem),&bT);
    clSetKernelArg(kMul,0,sizeof(cl_mem),&bT); clSetKernelArg(kMul,1,sizeof(cl_mem),&bO);

    size_t gws=8;
    cl_recording_qcom rec = pNew(rq, &err);
    printf("clNewRecordingQCOM err=%d\n", err);
    err  = clEnqueueNDRangeKernel(rq, kInc, 1, NULL, &gws, NULL, 0, NULL, NULL);
    err |= clEnqueueNDRangeKernel(rq, kMul, 1, NULL, &gws, NULL, 0, NULL, NULL);
    printf("record 2 dispatches err=%d\n", err);
    err = pEnd(rec);
    printf("clEndRecordingQCOM err=%d\n", err);

    // replay #1 (no updates): O = (A+1)*2 = {2,4,6,8,10,12,14,16}
    err = pEnq(q, rec, 0,NULL, 0,NULL, 0,NULL, 0,NULL, 0,NULL,NULL);
    clFinish(q);
    clEnqueueReadBuffer(q, bO, CL_TRUE, 0, sizeof(out), out, 0,NULL,NULL);
    int ok1 = (out[0]==2.0f && out[7]==16.0f);
    printf("replay#1 (no update) out={%.0f..%.0f} expect{2..16} -> %s\n", out[0], out[7], ok1?"OK":"WRONG");

    // replay #2 with ARG UPDATE: change dispatch 0 arg 0 (input) bA -> bA2  => O=(A2+1)*2={22..36}
    cl_array_arg_qcom upd = { .dispatch_index=0, .arg_index=0, .arg_size=sizeof(cl_mem), .arg_value=&bA2 };
    memset(out,0,sizeof(out));
    err = pEnq(q, rec, 1,&upd, 0,NULL, 0,NULL, 0,NULL, 0,NULL,NULL);
    printf("replay#2 (arg update) enqueue err=%d\n", err);
    clFinish(q);
    clEnqueueReadBuffer(q, bO, CL_TRUE, 0, sizeof(out), out, 0,NULL,NULL);
    int ok2 = (out[0]==22.0f && out[7]==36.0f);
    printf("replay#2 (input->A2) out={%.0f..%.0f} expect{22..36} -> %s\n", out[0], out[7], ok2?"OK":"WRONG");

    printf("VERDICT: %s\n", (ok1&&ok2) ? "SUPPORTED + FUNCTIONAL + ARG-UPDATE WORKS" : "PRESENT BUT BROKEN");
    pRel(rec);
    return 0;
}
