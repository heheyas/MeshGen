from huggingface_hub import hf_hub_download

repo = "heheyas/MeshGen"


texture_inpainter_path = hf_hub_download(repo, "texture_inpainter.pth")
shape_generator_path = hf_hub_download(repo, "shape_generator.pth")
pbr_decomposer_path = hf_hub_download(repo, "pbr_decomposer.pth")
mv_generator_path = hf_hub_download(repo, "mv_generator.pth")
