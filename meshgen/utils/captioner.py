import httpx
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

_client = None


## using qwen instead of chatGPT
def get_openai_client(backend):
    global _client
    if _client is None:
        if backend == "qwen":
            _client = OpenAI(
                api_key="",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif backend == "zhipu":
            _client = OpenAI(
                api_key="",
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                http_client=httpx.Client(trust_env=False),
            )
        elif backend == "openai":
            _client = OpenAI(
                api_key="",
                base_url="https://api.chatanywhere.tech/v1",
                http_client=httpx.Client(trust_env=False),
            )
    return _client


system_prompt = """
    You are an agent specialized in describing images rendered from 3D objects in English. You will be shown an image of a 3D object and asked to describe it with a short sentence in English. Please start with A .... Please respond in English.'
"""


def get_img_url(image_base64, quality):
    return {"url": f"data:image/jpeg;base64,{image_base64}", "detail": f"{quality}"}
    # return f"data:image/jpeg;base64,{image_base64}"


def analyze_image(img_url, backend="zhipu"):
    client = get_openai_client(backend)
    backend2model = {
        "qwen": "qwen-vl-max",
        "zhipu": "glm-4v",
        "openai": "gpt-4o-mini",
    }
    response = client.chat.completions.create(
        model=backend2model[backend],
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": img_url,
                    },
                ],
            },
        ],
        max_tokens=300,
        top_p=0.1,
    )

    return response.choices[0].message.content


def captioning(image_pil, quality="high", backend="openai"):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_url = get_img_url(img_str, quality)

    return analyze_image(img_url, backend)
