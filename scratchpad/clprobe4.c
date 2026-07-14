// QVAC-21623 Phase 2: authoritative cl_khr_command_buffer probe for Adreno 740.
// Advertised extension list LIES (R14) -> record+finalize+enqueue+replay a real kernel and verify.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    cl_platform_id plat; cl_device_id dev; cl_int err;
    if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS) { printf("no platform\n"); return 1; }
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS) { printf("no gpu\n"); return 1; }
    char name[256] = {0}; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("device: %s\n", name);

    size_t esz = 0; clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, NULL, &esz);
    char *exts = calloc(1, esz + 1); clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, esz, exts, NULL);
    printf("ADVERTISES cl_khr_command_buffer          : %s\n", strstr(exts, "cl_khr_command_buffer") ? "yes" : "no");
    printf("ADVERTISES ..._mutable_dispatch           : %s\n", strstr(exts, "cl_khr_command_buffer_mutable_dispatch") ? "yes" : "no");

    clCreateCommandBufferKHR_fn   pCreate = (clCreateCommandBufferKHR_fn)  clGetExtensionFunctionAddressForPlatform(plat, "clCreateCommandBufferKHR");
    clCommandNDRangeKernelKHR_fn  pND     = (clCommandNDRangeKernelKHR_fn) clGetExtensionFunctionAddressForPlatform(plat, "clCommandNDRangeKernelKHR");
    clFinalizeCommandBufferKHR_fn pFin    = (clFinalizeCommandBufferKHR_fn)clGetExtensionFunctionAddressForPlatform(plat, "clFinalizeCommandBufferKHR");
    clEnqueueCommandBufferKHR_fn  pEnq    = (clEnqueueCommandBufferKHR_fn) clGetExtensionFunctionAddressForPlatform(plat, "clEnqueueCommandBufferKHR");
    clReleaseCommandBufferKHR_fn  pRel    = (clReleaseCommandBufferKHR_fn) clGetExtensionFunctionAddressForPlatform(plat, "clReleaseCommandBufferKHR");
    void *pUpd = clGetExtensionFunctionAddressForPlatform(plat, "clUpdateMutableCommandsKHR");
    printf("RESOLVED Create=%p ND=%p Finalize=%p Enqueue=%p Update(mutable)=%p\n",
           (void*)pCreate, (void*)pND, (void*)pFin, (void*)pEnq, pUpd);
    if (!pCreate || !pND || !pFin || !pEnq) { printf("VERDICT: UNSUPPORTED (entry points unresolved)\n"); return 0; }

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    if (err != CL_SUCCESS) { printf("queue err=%d\n", err); return 1; }

    const char *src = "__kernel void inc(__global float*a,__global float*b){int i=get_global_id(0);b[i]=a[i]+1.0f;}";
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (clBuildProgram(prog, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) { printf("build fail\n"); return 1; }
    cl_kernel k = clCreateKernel(prog, "inc", &err);

    float in[8] = {0,1,2,3,4,5,6,7}, out[8] = {0};
    cl_mem A = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(in), in, &err);
    cl_mem B = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(out), NULL, &err);
    clSetKernelArg(k, 0, sizeof(cl_mem), &A);
    clSetKernelArg(k, 1, sizeof(cl_mem), &B);

    cl_command_buffer_khr cb = pCreate(1, &q, NULL, &err);
    printf("clCreateCommandBufferKHR err=%d\n", err);
    if (err != CL_SUCCESS) { printf("VERDICT: UNSUPPORTED (create failed)\n"); return 0; }
    size_t gws = 8;
    err = pND(cb, q, NULL, k, 1, NULL, &gws, NULL, 0, NULL, NULL, NULL);
    printf("clCommandNDRangeKernelKHR err=%d\n", err);
    err = pFin(cb);
    printf("clFinalizeCommandBufferKHR err=%d\n", err);
    err = pEnq(1, &q, cb, 0, NULL, NULL);
    printf("clEnqueueCommandBufferKHR err=%d\n", err);
    clFinish(q);
    clEnqueueReadBuffer(q, B, CL_TRUE, 0, sizeof(out), out, 0, NULL, NULL);
    int ok1 = (out[3] == 4.0f && out[7] == 8.0f);
    printf("run#1 out[3]=%.1f out[7]=%.1f (expect 4,8) -> %s\n", out[3], out[7], ok1 ? "OK" : "WRONG");

    // The whole point: replay the SAME finalized buffer without re-recording.
    memset(out, 0, sizeof(out));
    err = pEnq(1, &q, cb, 0, NULL, NULL); clFinish(q);
    clEnqueueReadBuffer(q, B, CL_TRUE, 0, sizeof(out), out, 0, NULL, NULL);
    int ok2 = (out[3] == 4.0f && out[7] == 8.0f);
    printf("replay out[3]=%.1f out[7]=%.1f -> %s\n", out[3], out[7], ok2 ? "OK" : "WRONG");

    printf("VERDICT: %s%s\n", (ok1 && ok2) ? "SUPPORTED + FUNCTIONAL" : "PRESENT BUT BROKEN",
           pUpd ? "  (mutable_dispatch resolvable)" : "  (no mutable_dispatch)");
    pRel(cb);
    return 0;
}
