
# 在Windows安裝Detectron2


---
## 目錄
[TOC]

---

## 請注意版本
```
Python 3.6 or higher
PyTorch 1.4 to 1.6
CUDA 9.2 or higher
Visual Studio 2013-2019
```

## 1. Create a conda environment
```
conda create -n mytorch python=3.6
conda activate mytorch
```

## 2. 安裝 Cuda
請至 Nvidia公司下載cuda
參照表格:
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
(這裡以cuda 10.1 為例)
下載:https://developer.nvidia.com/cuda-10.1-download-archive-base

## 安裝 pytorch

```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

版本一定要對好!!!

## 安裝Cython and Pycocotools
```
pip install cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

## 安裝 vs_BuildTools
依照你的cuda版本請對好
有的只支援到2017版
有的支援到2019版
請搜尋 版本對照表

vs_BuildTools安裝說明:
https://hjwang520.pixnet.net/blog/post/404280185-%E5%AE%89%E8%A3%9Dmicrosoft-visual-c%2B%2B-14.0




## 修改檔案
```
修改的第一個檔案:
 
  {your evn path}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
 
  example:
 
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h(190)
 
    static constexpr size_t DEPTH_LIMIT = 128;
 
      改成-->
 
    static const size_t DEPTH_LIMIT = 128;
 
修改的第二個檔案:
 
  {your evn path}\Lib\site-packages\torch\include\pybind11\cast.h
 
  example:
 
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\pybind11\cast.h(1449)
 
    explicit operator type&() { return *(this->value); }
 
      改成-->
 
    explicit operator type&() { return *((type*)this->value); }
 
```

## 修改檔案
打開你虛擬環境資料夾內
`/Lib/site-packages/torch/utils/cpp_extension.py`
的底下的`cpp_extension.py`的233行左右的程式碼

範例:
例如我的就在`C:\Users\用戶名\Anaconda3\envs\mytorch\Lib\site-packages\torch\utils`

```
match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode().strip())

修改為

match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(' cp950').strip())
```

## 下載 detectron2 windows版
https://github.com/LiaoZihZrong/detectron2-windows

下載後請將資料夾內的`setup.py`
約104行
```
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
            改成
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
                "-DWITH_CUDA",
            ]
```

==請注意這裡==
請用上面安裝完的`vs_BuildTools`
會有`Native Tools Command Prompt`
開啟 終端機

開啟剛剛建照的虛擬環境
`conda activate mytorch`

```
(並利用 cd指令 進入資料夾內)
cd detectron2
pip install -e .
```

## 理論上會出現錯誤資訊
此時很重要，請找到錯誤資訊
裡面會提到一個檔案有問題

`某路徑下的layers/csrc/nms_rotated/nms_rotated_cuda.cu(58)`

請你到該資料底下開啟 `nms_rotated_cuda.cu`
並更改開頭
改成如下:
```
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
//NOTE: replace relative import
/*#ifdef WITH_CUDA
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#endif
// TODO avoid this when pytorch supports "same directory" hipification
#ifdef WITH_HIP
#include "box_iou_rotated/box_iou_rotated_utils.h"
#endif*/
#include "box_iou_rotated/box_iou_rotated_utils.h"
```

然後重新下指令`pip install -e .`
此時安裝到一半
你可以重新整理 剛剛 
`某路徑下的layers/csrc/nms_rotated/nms_rotated_cuda.cu`
你會發現該檔案的開頭又變回原本的
這是你就要迅速將該程式碼
又改成
```
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
//NOTE: replace relative import
/*#ifdef WITH_CUDA
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#endif
// TODO avoid this when pytorch supports "same directory" hipification
#ifdef WITH_HIP
#include "box_iou_rotated/box_iou_rotated_utils.h"
#endif*/
#include "box_iou_rotated/box_iou_rotated_utils.h"
```
然後存檔

記住:
因為你下指令時，電腦會重新編譯一次
所以你改的東西才會不見
在程式編譯完後，終端機會顯示正在安裝中
會有一段等待期，你要在等待期把上述的程式碼改正!

預祝順利安裝完成!
