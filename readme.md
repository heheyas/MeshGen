## MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation

[Zilong Chen](https://heheyas.github.io), [Yikai Wang](), [Wenqiang Sun](), [Feng Wang](), [Yiwen Chen](), [Huaping Liu]()

Tsinghua University, BNU, HKUST, NTU


This repository contains the official implementation for MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation.


### Run locally
#### Install
First use `pip<24.1` since we are using an old version of lightning:
```bash
pip install 'pip<24.1'
```
If you are with CUDA 11:
Install `torch`:
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
then install other dependencies:
```bash
pip install -r requirements.txt
```

Or you are with CUDA 12:
Install `torch`:
```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 extra-index-url https://download.pytorch.org/whl/cu121
```
then install other dependencies:
```bash
pip install -r requirements_cuda12.txt
```

#### Shape Generation:
```bash
torchrun --nproc_per_node=<num-gpus> shapegen.py --images <image-dir> --output <output-dir>
```

#### Texture Generation:
```bash
torchrun --nproc_per_node=<num-gpus> texgen.py --meta <meta-file> --output <output-dir>
```

#### Textured Mesh Generation:
```bash
torchrun --nproc_per_node=<num-gpus> jointgen.py --images <output-dir> --output <output-dir>
```

#### Gradio demo:
```bash
python app.py
```

### Acknowledgement
- [Stable Diffusion]()
- [Paint3D]()
- [Zero123++]()
- [3DShape2Vecset]()

### Citation
```bibtex
@inproceedings{chen2025meshgen,
  author    = {Chen, Zilong and Wang, Yikai and Sun, Wenqiang and Wang, Feng and Chen, Yiwen and Liu, Huaping},
  title     = {MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025}
}
```

