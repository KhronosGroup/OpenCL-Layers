#include <stdlib.h>
#include <string.h>
#include <CL/cl_layer.h>

static struct _cl_icd_dispatch dispatch;

static const struct _cl_icd_dispatch *tdispatch;

CL_API_ENTRY cl_int CL_API_CALL
clGetLayerInfo(
    cl_layer_info  param_name,
    size_t         param_value_size,
    void          *param_value,
    size_t        *param_value_size_ret) {
  switch (param_name) {
  case CL_LAYER_API_VERSION:
    if (param_value) {
      if (param_value_size < sizeof(cl_layer_api_version))
        return CL_INVALID_VALUE;
      *((cl_layer_api_version *)param_value) = CL_LAYER_API_VERSION_100;
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(cl_layer_api_version);
    break;
  default:
    return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

static void _init_dispatch(void);

struct platform_icd {
  cl_platform_id pid;
  cl_uint        ngpus; /* number of GPU devices */
  cl_uint        ncpus; /* number of CPU devices */
  cl_uint        ndevs; /* total number of devices, of all types */
};

static int _cmp_platforms(const void *_a, const void *_b) {
  const struct platform_icd *a=(const struct platform_icd *)_a;
  const struct platform_icd *b=(const struct platform_icd *)_b;

  /* sort first platforms handling max gpu */
  if (a->ngpus > b->ngpus) return -1;
  if (a->ngpus < b->ngpus) return 1;
  /* sort next platforms handling max cpu */
  if (a->ncpus > b->ncpus) return -1;
  if (a->ncpus < b->ncpus) return 1;
  /* sort then platforms handling max devices */
  if (a->ndevs > b->ndevs) return -1;
  if (a->ndevs < b->ndevs) return 1;
  /* else consider platforms equal */
  return 0;
}

static struct platform_icd *_platforms = NULL;
static cl_uint _num_platforms = 0;
static cl_platform_id _default_id = NULL;

static void _count_devices(struct platform_icd *p) {
  tdispatch->clGetDeviceIDs(p->pid, CL_DEVICE_TYPE_GPU, 0, NULL, &(p->ngpus));
  tdispatch->clGetDeviceIDs(p->pid, CL_DEVICE_TYPE_CPU, 0, NULL, &(p->ncpus));
  tdispatch->clGetDeviceIDs(p->pid, CL_DEVICE_TYPE_ALL, 0, NULL, &(p->ndevs));
}

static void _sort_platforms(struct platform_icd *picds, cl_uint npicds) {
  if (npicds > 1) {
#ifndef _MSC_VER
    char* ocl_sort=getenv("OCL_ICD_PLATFORM_SORT");
    if (ocl_sort!=NULL && !strcmp(ocl_sort, "none")) {
#else
    #define ocl_sort_max_len 5
    char ocl_sort[ocl_sort_max_len];
    errno_t err = getenv_s(
      NULL,
      ocl_sort,
      (rsize_t)ocl_sort_max_len,
      "OCL_ICD_PLATFORM_SORT"
    );
    #undef ocl_sort_max_len
    if(err==0 && !strcmp(ocl_sort, "none")) {
#endif
      /* Platform not sorted */
      return;
    } else {
      for (cl_uint i = 0; i < npicds; i++) {
        _count_devices(picds + i);
      }
      qsort(picds, npicds, sizeof(*picds), &_cmp_platforms);
    }
  }
}

static void _set_default_id(void)
{
  long num_default_platform = 0;
#ifndef _MSC_VER
    const char *default_platform = getenv("OCL_ICD_DEFAULT_PLATFORM");
    if (default_platform) {
#else
    #define default_platform_max_len 5
    const char default_platform[default_platform_max_len] = "";
    errno_t err = getenv_s(
      NULL,
      (char*)default_platform,
      (rsize_t)default_platform_max_len,
      "OCL_ICD_DEFAULT_PLATFORM"
    );
    #undef default_platform_max_len
    if(err==0) {
#endif
    char *end_scan;
    num_default_platform = strtol(default_platform, &end_scan, 10);
    if (*default_platform == '\0' || *end_scan != '\0')
      return;
  }
  if (num_default_platform < 0 || num_default_platform >= (long)_num_platforms) return;
  _default_id = _platforms[num_default_platform].pid;
}

void _init_platforms(void)
{
  cl_int err;
  cl_platform_id *ids = NULL;

  err = tdispatch->clGetPlatformIDs(0, NULL, &_num_platforms);
  if (err != CL_SUCCESS)
    return;
  if (!_num_platforms)
    return;

  _platforms = (struct platform_icd *)calloc(
    _num_platforms,
    sizeof(struct platform_icd));
  if (!_platforms)
    return;

  ids = (cl_platform_id *)calloc(
    _num_platforms,
    sizeof(cl_platform_id));
  if (!ids)
    goto error;

  err = tdispatch->clGetPlatformIDs(_num_platforms, ids, NULL);
  if (err != CL_SUCCESS)
    goto error;

  for (cl_uint i =0; i < _num_platforms; i++)
    _platforms[i].pid = ids[i];
  _sort_platforms(_platforms, _num_platforms);
  _set_default_id();

  free(ids);
  return;
error:
  free(ids);
  free(_platforms);
}

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(
    cl_uint                         num_entries,
    const struct _cl_icd_dispatch  *target_dispatch,
    cl_uint                        *num_entries_out,
    const struct _cl_icd_dispatch **layer_dispatch_ret) {
  if (!target_dispatch || !layer_dispatch_ret ||!num_entries_out || num_entries < sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs))
    return CL_INVALID_VALUE;

  tdispatch = target_dispatch;
  _init_platforms();
  _init_dispatch();

  *layer_dispatch_ret = &dispatch;
  *num_entries_out = sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs);
  return CL_SUCCESS;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs_wrap(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms)
{
  if (platforms == NULL && num_platforms == NULL)
    return CL_INVALID_VALUE;
  if (num_entries == 0 && platforms != NULL)
    return CL_INVALID_VALUE;
  if (_num_platforms == 0) {
    if ( num_platforms != NULL )
      *num_platforms = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
  }
  if (num_platforms != NULL)
    *num_platforms = _num_platforms;
  if (platforms != NULL) {
    cl_uint n_platforms = _num_platforms < num_entries ? _num_platforms : num_entries;
    for( cl_uint i = 0; i < n_platforms; i++)
      platforms[i] = _platforms[i].pid;
  }
  return CL_SUCCESS;
}

static inline int _check_platform(cl_platform_id pid) {
  int good=0;
  for(cl_uint j = 0; j < _num_platforms; j++) {
    if( _platforms[j].pid == pid) {
      good=1;
      break;
    }
  }
  return good;
}

static CL_API_ENTRY cl_context CL_API_CALL clCreateContext_wrap(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
  if (properties != NULL){
    cl_uint i = 0;
    while (properties[i] != 0) {
      if (properties[i] == CL_CONTEXT_PLATFORM) {
        if ((struct _cl_platform_id *) properties[i+1] == NULL) {
          goto out;
        } else {
          if (!_check_platform((cl_platform_id) properties[i+1])) {
            goto out;
          }
        }
        break;
      }
      i += 2;
    }
  }
  return tdispatch->clCreateContext(
    properties,
    num_devices,
    devices,
    pfn_notify,
    user_data,
    errcode_ret);
out:
  *errcode_ret = CL_INVALID_PLATFORM;
  return NULL;
}

static CL_API_ENTRY cl_context CL_API_CALL clCreateContextFromType_wrap(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
  if (_num_platforms == 0)
    goto out;

  if (properties != NULL){
    cl_uint i = 0;
    while (properties[i] != 0) {
      if (properties[i] == CL_CONTEXT_PLATFORM) {
        if ((struct _cl_platform_id *) properties[i+1] == NULL) {
          goto out;
        } else {
          if (!_check_platform((cl_platform_id) properties[i+1])) {
            goto out;
          }
        }
        return tdispatch->clCreateContextFromType(
          properties,
          device_type,
          pfn_notify,
          user_data,
          errcode_ret);
      }
      i += 2;
    }
  } else {
    if (_default_id == NULL)
      goto out;
    cl_context_properties new_props[3];
    new_props[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;
    new_props[1] = (cl_context_properties)_default_id;
    new_props[2] = (cl_context_properties)NULL;
    return tdispatch->clCreateContextFromType(
      new_props,
      device_type,
      pfn_notify,
      user_data,
      errcode_ret);
  }
out:
  *errcode_ret = CL_INVALID_PLATFORM;
  return NULL;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetGLContextInfoKHR_wrap(
    const cl_context_properties *  properties,
    cl_gl_context_info             param_name,
    size_t                         param_value_size,
    void *                         param_value,
    size_t *                       param_value_size_ret) {
  if (properties != NULL){
    cl_uint i = 0;
    while (properties[i] != 0) {
      if (properties[i] == CL_CONTEXT_PLATFORM) {
        if ((struct _cl_platform_id *) properties[i+1] == NULL) {
          goto out;
        } else {
          if (!_check_platform((cl_platform_id) properties[i+1])) {
            goto out;
          }
        }
        break;
      }
      i += 2;
    }
  }
  return tdispatch->clGetGLContextInfoKHR(
    properties,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
out:
  return CL_INVALID_PLATFORM;
}

static inline
cl_platform_id _select_platform_id(cl_platform_id pid) {
  if (pid) return pid;
  return _default_id;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
    cl_platform_id    platform,
    cl_platform_info  param_name,
    size_t            param_value_size,
    void *            param_value,
    size_t *          param_value_size_ret) {
  platform = _select_platform_id(platform);
  return tdispatch->clGetPlatformInfo(
    platform,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs_wrap(
    cl_platform_id    platform,
    cl_device_type    device_type,
    cl_uint           num_entries,
    cl_device_id *    devices,
    cl_uint *         num_devices) {
  platform = _select_platform_id(platform);
  return tdispatch->clGetDeviceIDs(
    platform,
    device_type,
    num_entries,
    devices,
    num_devices);
}

static CL_API_ENTRY cl_int CL_API_CALL clUnloadPlatformCompiler_wrap(
    cl_platform_id  platform){
  platform = _select_platform_id(platform);
  return tdispatch->clUnloadPlatformCompiler(platform);
}

static CL_API_ENTRY void * CL_API_CALL clGetExtensionFunctionAddressForPlatform_wrap(
    cl_platform_id  platform,
    const char *    func_name){
  platform = _select_platform_id(platform);
  return tdispatch->clGetExtensionFunctionAddressForPlatform(
    platform,
    func_name);
}

static void _init_dispatch(void) {
  dispatch.clGetPlatformIDs = &clGetPlatformIDs_wrap;
  dispatch.clGetPlatformInfo = &clGetPlatformInfo_wrap;
  dispatch.clGetDeviceIDs = &clGetDeviceIDs_wrap;
  dispatch.clCreateContext = &clCreateContext_wrap;
  dispatch.clCreateContextFromType = &clCreateContextFromType_wrap;
  dispatch.clGetGLContextInfoKHR = &clGetGLContextInfoKHR_wrap;
  dispatch.clUnloadPlatformCompiler = &clUnloadPlatformCompiler_wrap;
  dispatch.clGetExtensionFunctionAddressForPlatform = &clGetExtensionFunctionAddressForPlatform_wrap;
}
