# Local ROCm development

To work with kvpress on workstation supporting ROCm. 
First of all, apply [the patch](Support_ROCm_pytorch.patch).
Then verify in a `uv run python` session:

```python
  import torch

  # Shows the full version string (often includes +rocm6.x or +cu12x)
  print(torch.__version__)

  # ROCm build sets this; None if not ROCm
  print(torch.version.hip)

  # CUDA build sets this; None if not CUDA
  print(torch.version.cuda)

  # Whether GPU acceleration is actually available
  print(torch.cuda.is_available()) 

  # Device name
  print(torch.cuda.get_device_name(0))

  # Device properties
  print(torch.cuda.get_device_properties(0))
```

your output should be something like:

```bash
2.7.1+rocm6.2.4

6.2.41134-65d174c3e

None

True
                                           
AMD Radeon Graphics
                                     
_CudaDeviceProperties(name='AMD Radeon Graphics', major=11, minor=0,               
gcnArchName='gfx1100', total_memory=16368MB, multi_processor_count=30,             
uuid=65346161-3739-3838-3061-336432616431, L2_cache_size=4MB)
```

Also it is good to verify the [rocminfo](rocminfo.txt) and `rocm-smi`.
