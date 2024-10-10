import os
import gradio as gr
import numpy as np
import torch
import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import torch.distributed as dist
from einops import repeat
from PIL import Image
import json
import tempfile
from uuid import uuid4
from functools import partial

# import spaces

from meshgen.util import instantiate_from_config
from meshgen.utils.io import write_video, export_mesh, write_image
from meshgen.utils.render import (
    render_mesh_spiral_offscreen,
)
from meshgen.utils.images import preprocess_image
from meshgen.utils.remesh import pyacvd_remesh
from meshgen.utils.hf_weights import (
    shape_generator_path,
    pbr_decomposer_path,
    texture_inpainter_path,
    mv_generator_path,
)
from meshgen.utils.misc import set_if_none

texture_generator_tempdir = tempfile.TemporaryDirectory()

shape_generator_config = OmegaConf.load("configs/shapegen.yaml")
shape_generator_config.params.ckpt_path = shape_generator_path
shape_generator = instantiate_from_config(shape_generator_config).to("cuda")

texture_generator_config = OmegaConf.load("configs/texgen.yaml")
texture_generator_config.params.use_pbr = False
set_if_none(
    texture_generator_config.params,
    "exp_dir",
    texture_generator_tempdir.name,
)
set_if_none(
    texture_generator_config.params.multiview_generator.params,
    "ckpt_path",
    mv_generator_path,
)
set_if_none(
    texture_generator_config.params.pbr_decomposer.params,
    "ckpt_path",
    pbr_decomposer_path,
)
texture_generator_config.params.texture_inpainter = None
# set_if_none(
#     texture_generator_config.params.texture_inpainter.params,
#     "ckpt_path",
#     texture_inpainter_path,
# )
texture_generator = instantiate_from_config(texture_generator_config).to("cuda")

STYLE = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <style>
        .alert, .alert div, .alert b {
            color: black !important;
        }
    </style>
"""

ICONS = {
    "info": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#0d6efd" class="bi bi-info-circle-fill flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>
    </svg>""",
    "cursor": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#0dcaf0" class="bi bi-hand-index-thumb-fill flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M8.5 1.75v2.716l.047-.002c.312-.012.742-.016 1.051.046.28.056.543.18.738.288.273.152.456.385.56.642l.132-.012c.312-.024.794-.038 1.158.108.37.148.689.487.88.716.075.09.141.175.195.248h.582a2 2 0 0 1 1.99 2.199l-.272 2.715a3.5 3.5 0 0 1-.444 1.389l-1.395 2.441A1.5 1.5 0 0 1 12.42 16H6.118a1.5 1.5 0 0 1-1.342-.83l-1.215-2.43L1.07 8.589a1.517 1.517 0 0 1 2.373-1.852L5 8.293V1.75a1.75 1.75 0 0 1 3.5 0z"/>
    </svg>""",
    "wait": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#6c757d" class="bi bi-hourglass-split flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M2.5 15a.5.5 0 1 1 0-1h1v-1a4.5 4.5 0 0 1 2.557-4.06c.29-.139.443-.377.443-.59v-.7c0-.213-.154-.451-.443-.59A4.5 4.5 0 0 1 3.5 3V2h-1a.5.5 0 0 1 0-1h11a.5.5 0 0 1 0 1h-1v1a4.5 4.5 0 0 1-2.557 4.06c-.29.139-.443.377-.443.59v.7c0 .213.154.451.443.59A4.5 4.5 0 0 1 12.5 13v1h1a.5.5 0 0 1 0 1h-11zm2-13v1c0 .537.12 1.045.337 1.5h6.326c.216-.455.337-.963.337-1.5V2h-7zm3 6.35c0 .701-.478 1.236-1.011 1.492A3.5 3.5 0 0 0 4.5 13s.866-1.299 3-1.48V8.35zm1 0v3.17c2.134.181 3 1.48 3 1.48a3.5 3.5 0 0 0-1.989-3.158C8.978 9.586 8.5 9.052 8.5 8.351z"/>
    </svg>""",
    "done": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#198754" class="bi bi-check-circle-fill flex-shrink-0 me-2" viewBox="0 0 16 16">
    <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z"/>
    </svg>""",
}

icons2alert = {
    "info": "primary",  # blue
    "cursor": "info",  # light blue
    "wait": "secondary",  # gray
    "done": "success",  # green
}

render_kwargs = {
    "num_frames": 90,
    "elevation": 0,
    "radius": 2.0,
    "rotate": False,
    "force_new_render": True,
}

update_guide = lambda GUIDE_TEXT, icon_type="info": gr.HTML(
    value=message(GUIDE_TEXT, icon_type)
)


def message(text, icon_type="info"):
    return f"""{STYLE}  <div class="alert alert-{icons2alert[icon_type]} d-flex align-items-center" role="alert"> {ICONS[icon_type]}
                            <div> 
                                {text} 
                            </div>
                        </div>"""


def update_submit_button_interactivity(processed_image, mesh):
    re_remove = False
    shapegen = False
    texgen = False
    onestop = False
    if processed_image is not None:
        re_remove = True
        shapegen = True
        onestop = True

    if mesh is not None:
        texgen = True

    return (
        gr.Button(interactive=re_remove),
        gr.Button(interactive=shapegen),
        gr.Button(interactive=texgen),
        gr.Button(interactive=onestop),
    )


def do_preprocess_image(image, border_ratio, ignore_alpha, backend="birefnet"):
    if image is None:
        return None
    preprocessed = preprocess_image(
        image,
        size=512,
        border_ratio=border_ratio,
        remove_bg=True,
        ignore_alpha=ignore_alpha,
        backend=backend,
    )

    return preprocessed


def do_shapegen(
    image,
    cfg,
    n_steps,
    seed,
    do_remesh,
    precision,
    thresh=0.5,
    R=384,
    target_num_faces=20000,
    flip=False,
):
    with torch.inference_mode():
        dtype = torch.float32 if precision == "fp32" else torch.float16
        cond = np.array(image.convert("RGB"))
        uncond = np.zeros_like(cond)
        with torch.autocast("cuda", dtype):
            denoised = shape_generator.sample_one(
                cond,
                uncond,
                n_samples=1,
                cfg=cfg,
                n_steps=n_steps,
                seed=seed,
            )
            v, f = shape_generator.decode_shape(denoised[0:1], thresh=thresh, R=R)

            if do_remesh:
                v, f = pyacvd_remesh(v, f, target_num_faces=target_num_faces)

            if flip:
                f = f[..., [0, 2, 1]]

    frames = render_mesh_spiral_offscreen(v, f, **render_kwargs)
    temp_dir = Path(texture_generator_tempdir.name) / uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)
    export_mesh(v, f, temp_dir / "mesh.obj")
    write_video(temp_dir / "spiral.mp4", frames)

    return str(temp_dir / "mesh.obj"), str(temp_dir / "spiral.mp4"), str(temp_dir)
    # return str(temp_dir / "mesh.obj"), str(temp_dir / "spiral.mp4")


def texgen_mv_shaded(image, mesh):
    return texture_generator.multiview_shaded_gen(
        mesh, image, verbose=False, skip_captioning=True
    )


def texgen_pbr_decompose(mv_shaded, ip):
    return texture_generator.pbr_decompose(mv_shaded, ip, verbose=False)


def texgen_shaded_bp(mv_shaded, mesh, exp_dir, export_dir):
    return texture_generator.texture_bp(
        mesh, mv_shaded, exp_dir, export_dir, verbose=False
    )


_TITLE = """MeshGen"""

_HEADER_ = """
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/heheyas/MeshGen' target='_blank'><b>MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation</b></a></h2>

**MeshGen** is a 3D native diffusion model trained with render-enhanced auto-encoder and generative data augmentation.
Code: <a href='https://github.com/heheyas/MeshGen' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/xxxx.xxxx' target='_blank'>ArXiv</a>.
"""

_CITE_ = r"""
If MeshGen is helpful, please help to ⭐ the <a href='https://github.com/heheyas/MeshGen' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/heheyas/MeshGen?style=social)](https://github.com/heheyas/MeshGen)
---
📝 **Citation**
If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{chen2024meshgen,
  title={MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation},
  author={Chen, Zilong and Wang, Yikai and Sun, Wenqiang and Wang, Feng and Liu, Huaping},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```
📋 **License**
MIT LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/heheyas/MeshGen/blob/main/LICENSE) for details.
📧 **Contact**
If you have any questions, feel free to open a discussion or contact Zilong at <b>jaysonabcchen@gmail.com</b>.
"""

with gr.Blocks(title="MeshGen Demo", css="style.css") as demo:
    this_mesh = gr.State()
    this_ip = gr.State()
    this_tempdir = gr.State()

    with gr.Row():
        gr.Markdown(_HEADER_)
    with gr.Row():
        guide_text_i2m = gr.HTML(message("Please input an image!"), visible=True)
    # with gr.Row(variant="panel"):
    # shapegen
    with gr.Row():
        with gr.Column():
            # images
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image",
                    image_mode="RGB",
                    type="pil",
                    interactive=True,
                )

        with gr.Column():
            with gr.Row():
                output_model_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    interactive=False,
                )
                output_spiral = gr.Video(label="Orbit Video")
            with gr.Row(variant="panel"):
                gr.Markdown(
                    """Try a different <b>seed</b> if the result is unsatisfying (Default: 42)."""
                )

    # texgen
    with gr.Row():
        with gr.Column():
            this_mv_shaded = gr.Image(
                label="Multiview Shaded",
                image_mode="RGB",
                type="pil",
                interactive=True,
            )
        with gr.Column():
            albedo = gr.Image(label="Albedo", image_mode="RGB", type="pil")
        with gr.Column():
            roughness = gr.Image(label="Roughness", image_mode="RGB", type="pil")
        with gr.Column():
            metalic = gr.Image(label="Metallic", image_mode="RGB", type="pil")
    with gr.Row():
        with gr.Column():
            textured_spiral = gr.Video(
                label="Textured Orbit Video", sources=["upload"], height=300
            )
        with gr.Column():
            textured_model_glb = gr.Model3D(
                label="Textured Model (GLB Format)", interactive=False
            )

    # options and examples
    with gr.Row():
        with gr.Column():
            # detailed options
            with gr.Row():
                with gr.Group():
                    with gr.Accordion("Detailed options", open=False):
                        ignore_alpha = gr.Checkbox(
                            value=True,
                            label="Ignore alpha channel of the input image",
                            info="If set to False, the background removal will be skipped, and only the recenter will be performed.",
                        )
                        rembg_backend = gr.Radio(
                            ["rembg", "bria", "birefnet"],
                            value="birefnet",
                            label="Background removel backend",
                        )
                        border_ratio = gr.Slider(
                            value=0.15,
                            minimum=0.1,
                            maximum=0.5,
                            step=0.05,
                            label="Border ratio for clipped image",
                        )
                        sample_seed = gr.Number(value=42, label="seed", precision=0)
                        sample_steps = gr.Slider(
                            label="Number of sampling steps",
                            minimum=30,
                            maximum=75,
                            value=75,
                            step=5,
                        )
                        cfg_scale = gr.Slider(
                            value=3.0,
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            label="Scale for classifier-free guidance",
                        )
                        n_steps = gr.Slider(
                            value=25,
                            minimum=25,
                            maximum=100,
                            step=1,
                            label="Number of inference steps",
                        )
                        do_remesh = gr.Checkbox(
                            value=True, label="Remesh the generated mesh"
                        )
                        R = gr.Slider(
                            value=384,
                            minimum=256,
                            maximum=512,
                            step=1,
                            label="Resolution for marching cubes",
                        )
                        thresh = gr.Slider(
                            value=0.5,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            label="Threshold for marching cubes",
                        )
                        target_num_faces = gr.Slider(
                            value=20000,
                            minimum=1000,
                            maximum=50000,
                            step=1000,
                            label="Target number of faces for remeshing",
                        )
                        flip = gr.Checkbox(
                            value=False,
                            label="Flip the generated mesh, set to True if the mesh is upside down",
                        )
                        precision = gr.Radio(
                            value="fp16",
                            label="Precision",
                            choices=["fp32", "fp16"],
                        )

            # buttons
            with gr.Row():
                redo_remove_bg = gr.Button(
                    "Re-remove background",
                    elem_id="redo_remove_bg",
                    variant="primary",
                    interactive=False,
                )
                shapegen_submit = gr.Button(
                    "Generate shape",
                    elem_id="shape_generate",
                    variant="primary",
                    interactive=False,
                )
                texgen_submit = gr.Button(
                    "Generate texture",
                    elem_id="texture_generate",
                    variant="primary",
                    interactive=False,
                )
                one_stop_submit = gr.Button(
                    "One stop",
                    elem_id="one_stop",
                    variant="primary",
                    interactive=False,
                )

            # examples
            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("assets/examples", img_name)
                        for img_name in sorted(os.listdir("assets/examples"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=14,
                )
    gr.Markdown(_CITE_)

    # mv_images = gr.State()
    input_image.change(
        fn=update_submit_button_interactivity,
        inputs=[processed_image, output_model_obj],
        outputs=[redo_remove_bg, shapegen_submit, texgen_submit, one_stop_submit],
        queue=False,
    ).success(
        fn=partial(update_guide, "Preprocessing the image!", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=do_preprocess_image,
        inputs=[input_image, border_ratio, ignore_alpha, rembg_backend],
        outputs=[processed_image],
        queue=True,
    ).success(
        fn=partial(
            update_guide,
            "Click <b>Generate shape</b> to generate mesh! Click <b>One stop</b> to generate shape and texture at once.",
            "cursor",
        ),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=update_submit_button_interactivity,
        inputs=[processed_image, output_model_obj],
        outputs=[redo_remove_bg, shapegen_submit, texgen_submit, one_stop_submit],
        queue=False,
    )

    redo_remove_bg.click(
        fn=update_submit_button_interactivity,
        inputs=[processed_image, output_model_obj],
        outputs=[redo_remove_bg, shapegen_submit, texgen_submit, one_stop_submit],
        queue=False,
    ).success(
        fn=do_preprocess_image,
        inputs=[input_image, border_ratio, ignore_alpha, rembg_backend],
        outputs=[processed_image],
        queue=True,
    ).success(
        fn=update_submit_button_interactivity,
        inputs=[processed_image, output_model_obj],
        outputs=[redo_remove_bg, shapegen_submit, texgen_submit, one_stop_submit],
        queue=False,
    )

    shapegen_submit.click(
        fn=partial(update_guide, "Generating mesh!", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=do_shapegen,
        inputs=[
            processed_image,
            cfg_scale,
            n_steps,
            sample_seed,
            do_remesh,
            precision,
            thresh,
            R,
            target_num_faces,
            flip,
        ],
        outputs=[output_model_obj, output_spiral, this_tempdir],
        # outputs=[output_model_obj, output_spiral],
        queue=True,
    ).success(
        fn=partial(update_guide, "Mesh generated successfully!", "done"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=update_submit_button_interactivity,
        inputs=[processed_image, output_model_obj],
        outputs=[redo_remove_bg, shapegen_submit, texgen_submit, one_stop_submit],
        queue=False,
    )

    texgen_submit.click(
        fn=partial(update_guide, "Generating texture!", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=partial(update_guide, "Generating shaded multi-views", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=texgen_mv_shaded,
        inputs=[processed_image, output_model_obj],
        outputs=[this_ip, this_mv_shaded, this_mesh],
        queue=True,
    ).success(
        fn=partial(update_guide, "Decomposing PBR channels", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=texgen_pbr_decompose,
        inputs=[this_mv_shaded, this_ip],
        outputs=[albedo, roughness, metalic],
        queue=True,
    ).success(
        fn=partial(update_guide, "Back-projecting multi-views", "done"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=texgen_shaded_bp,
        inputs=[this_mv_shaded, this_mesh, this_tempdir, this_tempdir],
        outputs=[textured_model_glb, textured_spiral],
        queue=True,
    ).success(
        fn=partial(update_guide, "Texture generated successfully!", "done"),
        outputs=[guide_text_i2m],
        queue=False,
    )

    one_stop_submit.click(
        fn=partial(update_guide, "Generating mesh!", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=do_shapegen,
        inputs=[
            processed_image,
            cfg_scale,
            n_steps,
            sample_seed,
            do_remesh,
            precision,
            thresh,
            R,
            target_num_faces,
            flip,
        ],
        outputs=[output_model_obj, output_spiral, this_tempdir],
        # outputs=[output_model_obj, output_spiral],
        queue=True,
    ).success(
        fn=partial(update_guide, "Mesh generated successfully!", "done"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=update_submit_button_interactivity,
        inputs=[processed_image, output_model_obj],
        outputs=[redo_remove_bg, shapegen_submit, texgen_submit, one_stop_submit],
        queue=False,
    ).success(
        fn=partial(update_guide, "Generating texture!", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=partial(update_guide, "Generating shaded multi-views", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=texgen_mv_shaded,
        inputs=[processed_image, output_model_obj],
        outputs=[this_ip, this_mv_shaded, this_mesh],
        queue=True,
    ).success(
        fn=partial(update_guide, "Decomposing PBR channels", "wait"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=texgen_pbr_decompose,
        inputs=[this_mv_shaded, this_ip],
        outputs=[albedo, roughness, metalic],
        queue=True,
    ).success(
        fn=partial(update_guide, "Back-projecting multi-views", "done"),
        outputs=[guide_text_i2m],
        queue=False,
    ).success(
        fn=texgen_shaded_bp,
        inputs=[this_mv_shaded, this_mesh, this_tempdir, this_tempdir],
        outputs=[textured_model_glb, textured_spiral],
        queue=True,
    ).success(
        fn=partial(update_guide, "Texture generated successfully!", "done"),
        outputs=[guide_text_i2m],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(server_port=7077, debug=True)

    texture_generator_tempdir.cleanup()
