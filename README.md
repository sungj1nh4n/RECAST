# Speaking ITM’s Language

RECAST selects evidence frames for long-video multiple-choice QA. A routed Qwen pipeline converts the question and four options into one visual description. Frozen BLIP scores each candidate frame against that description. Rosin thresholding filters the score curve and temporal MMR fills the frame budget.

## Try the selector

Run from this directory with Python 3.10+ in a virtual environment.

```sh
python -m pip install -r requirements.txt
python - <<'PY'
from src.model import select_frames
scores = [0.01] * 30
scores[2], scores[3], scores[24] = 0.97, 0.96, 0.85
print(select_frames(scores, k=2, relevance=0.3, sigma=5).tolist())
PY
```

The output is `[2, 24]`. The second-highest score lies only one second from the best frame. Equation 1 favors the later evidence at frame 24. The [walkthrough](notebooks/walkthrough.ipynb) runs on CPU with synthetic inputs.

## Run with pretrained models

Provide chronological RGB PIL images sampled at one frame per second, a question, and its four options. The chat endpoint must serve the three Qwen2.5-Instruct model IDs used in §IV-A5. Set `QWEN_BASE_URL` to the API base including its version prefix. BLIP uses its checkpoint processor and downloads weights on first use. Reuse both adapters across questions.

```python
import os
from pathlib import Path
import yaml
from src.backends import blip_scorer, qwen_http
from src.model import recast

settings = yaml.safe_load(Path("configs/base.yaml").read_text())
relevance, sigma = settings["benchmarks"]["LongVideoBench"]
generate = qwen_http(os.environ["QWEN_BASE_URL"], os.getenv("QWEN_API_KEY"))
score = blip_scorer()
# frames, question and options come from the caller.
description, scores, indices = recast(
    question, options, frames, generate, score,
    k=settings["budgets"]["Qwen2-VL-7B"], relevance=relevance, sigma=sigma,
)
selected_frames = [frames[i] for i in indices.tolist()]
```

The QA backbone receives the selected frames in chronological order with the original question and options. For cached scores, call `select_frames` directly with one probability per original candidate. Removing low-scoring rows before calling it changes distances in Equation 1.

## Implementation map

| Paper | File |
|---|---|
| §III-B, Fig. 2 — routing, constrained synthesis, one correction | [src/reformulation.py](src/reformulation.py) |
| §III-C, Eq. 1 — histogram cut, greedy temporal selection and composition | [src/model.py](src/model.py) |
| §IV-A — Qwen and BLIP inference | [src/backends.py](src/backends.py) |
| Table 2 — dataset parameters and model budgets | [configs/base.yaml](configs/base.yaml) |
| Unpublished details and local choices | [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) |

Sungjin Han, Thang Vu and Junyeong Kim. “Speaking ITM’s Language: Query Reformulation and Temporal-MMR for Long-Video QA.” IEEE Access 14, 2026. [Paper](https://doi.org/10.1109/ACCESS.2026.3725764)
