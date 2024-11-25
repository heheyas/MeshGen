## MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation


This repository contains the official implementation for MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation.


### Demos:
Colab:
Huggingface:

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
torchrun --nproc_per_node=<> shapegen.py --images <> --output <>
```

#### Texture Generation:
```bash
torchrun --nproc_per_node=<> texgen.py
```

#### Textured Mesh Generation:
```bash
torchrun --nproc_per_node=<>
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
@article{chen2024meshgen,
    title={MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation},
    author={Chen, Zilong and Wang, Yikai and Sun, Wenqiang and Wang, Feng and Liu, Huaping},
    journal={},
    year={2024}
}
```

