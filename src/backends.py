"""Pretrained inference adapters for §IV-A3, §IV-A5 and Table 8."""
import json
from urllib.request import Request, urlopen

import torch


def qwen_http(base_url, api_key=None):
    """Return a generator for a chat-completions endpoint serving all three checkpoints."""
    # §IV-A5, Fig. 2. Token caps are [UNSPECIFIED] and reconstructed here.
    settings = dict(route=(3, 64), ground=(7, 256), decompose=(7, 256), synthesize=(14, 256))

    def generate(stage, instruction, payload):
        size, cap = settings[stage]
        body = dict(model=f'Qwen/Qwen2.5-{size}B-Instruct', temperature=0, max_tokens=cap,
                    messages=[dict(role='system', content=instruction),
                              dict(role='user', content=json.dumps(payload, ensure_ascii=False))])
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        request = Request(base_url.rstrip('/') + '/chat/completions',
                          data=json.dumps(body).encode(), headers=headers)
        with urlopen(request, timeout=180) as response:
            return json.load(response)['choices'][0]['message']['content']

    return generate


def blip_scorer(device=None):
    """Load the frozen BLIP ITM once and return score(frames, description), §IV-A3."""
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    name = 'Salesforce/blip-itm-large-coco'
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForImageTextRetrieval.from_pretrained(name).to(device=device, dtype=dtype)
    model.eval().requires_grad_(False)

    @torch.inference_mode()
    def score(frames, description):
        batches = []
        for start in range(0, len(frames), 24):  # Table 8
            batch = list(frames[start:start + 24])
            inputs = processor(images=batch, text=[description] * len(batch),
                               return_tensors='pt', padding=True).to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype)  # [B,3,384,384]
            logits = model(**inputs, use_itm_head=True).itm_score  # [B,2]
            batches.append(logits.float().softmax(-1)[:, 1].cpu())  # [B]
        return torch.cat(batches) if batches else torch.empty(0)  # [N]

    return score
