#!/usr/bin/env python3
"""Send an OpenAI-compatible multimodal request directly to the vLLM server."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "vlx-seek-1.5-3b-sanitized"
DEFAULT_IMAGE_URL = "/workspace/omchat/demo_1920_1080.jpeg"
DEFAULT_QUESTION = "检测图像中的枕头和鞋子和柜子，生成对应的标签并返回物体的索引"
DEFAULT_BBOX_LIST = [
    [412.0, 450.0, 589.0, 771.0],
    [501.0, 639.0, 1258.0, 1020.0],
    [197.0, 745.0, 381.0, 884.0],
    [1160.0, 196.0, 1298.0, 390.0],
    [1247.0, 701.0, 1391.0, 846.0],
    [1506.0, 685.0, 1870.0, 1078.0],
    [759.0, 675.0, 876.0, 724.0],
    [766.0, 579.0, 856.0, 638.0],
    [1019.0, 214.0, 1144.0, 399.0],
    [892.0, 230.0, 1006.0, 405.0],
    [1046.0, 641.0, 1221.0, 745.0],
    [608.0, 307.0, 895.0, 770.0],
    [403.0, 941.0, 495.0, 1019.0],
    [1292.0, 637.0, 1349.0, 706.0],
    [475.0, 943.0, 518.0, 1030.0],
    [952.0, 679.0, 1060.0, 721.0],
    [794.0, 629.0, 833.0, 682.0],
    [1026.0, 659.0, 1174.0, 735.0],
    [261.0, 219.0, 460.0, 807.0],
    [1531.0, 629.0, 1606.0, 689.0],
    [764.0, 1020.0, 873.0, 1079.0],
    [888.0, 650.0, 1026.0, 714.0],
    [494.0, 614.0, 552.0, 652.0],
    [890.0, 640.0, 1221.0, 742.0],
    [454.0, 617.0, 501.0, 655.0],
    [401.0, 938.0, 519.0, 1030.0],
    [903.0, 639.0, 1052.0, 664.0],
    [605.0, 668.0, 668.0, 772.0],
    [869.0, 1013.0, 985.0, 1078.0],
    [90.0, 682.0, 250.0, 860.0],
    [453.0, 616.0, 552.0, 656.0],
    [887.0, 638.0, 1051.0, 712.0],
    [89.0, 228.0, 273.0, 677.0],
    [765.0, 580.0, 856.0, 686.0],
    [509.0, 551.0, 538.0, 575.0],
    [889.0, 194.0, 1300.0, 406.0],
    [91.0, 837.0, 520.0, 978.0],
    [453.0, 496.0, 495.0, 524.0],
    [426.0, 650.0, 578.0, 740.0],
    [520.0, 496.0, 541.0, 524.0],
    [448.0, 556.0, 474.0, 586.0],
    [490.0, 678.0, 523.0, 722.0],
    [760.0, 1012.0, 987.0, 1079.0],
    [0.0, 2.0, 89.0, 1078.0],
    [822.0, 425.0, 894.0, 478.0],
    [423.0, 648.0, 583.0, 772.0],
    [547.0, 676.0, 567.0, 704.0],
    [471.0, 555.0, 495.0, 588.0],
    [428.0, 622.0, 458.0, 653.0],
    [86.0, 821.0, 1505.0, 1079.0],
    [448.0, 651.0, 551.0, 734.0],
    [1510.0, 728.0, 1550.0, 1080.0],
    [436.0, 548.0, 466.0, 582.0],
    [480.0, 657.0, 544.0, 735.0],
    [788.0, 617.0, 838.0, 682.0],
    [518.0, 470.0, 543.0, 524.0],
    [180.0, 6.0, 1916.0, 1076.0],
    [722.0, 638.0, 1257.0, 820.0],
    [1156.0, 734.0, 1256.0, 813.0],
    [510.0, 470.0, 547.0, 590.0],
    [426.0, 491.0, 466.0, 520.0],
    [515.0, 492.0, 542.0, 525.0],
    [508.0, 490.0, 545.0, 574.0],
    [886.0, 638.0, 1258.0, 821.0],
    [441.0, 548.0, 488.0, 587.0],
    [555.0, 459.0, 588.0, 720.0],
    [518.0, 479.0, 544.0, 572.0],
    [549.0, 2.0, 1620.0, 838.0],
    [250.0, 843.0, 444.0, 902.0],
    [1046.0, 643.0, 1220.0, 679.0],
    [1508.0, 681.0, 1606.0, 1077.0],
    [451.0, 467.0, 473.0, 502.0],
    [645.0, 707.0, 1250.0, 827.0],
    [90.0, 167.0, 442.0, 279.0],
    [405.0, 0.0, 842.0, 79.0],
    [416.0, 458.0, 580.0, 654.0],
    [426.0, 656.0, 474.0, 736.0],
    [541.0, 649.0, 581.0, 730.0],
    [754.0, 1012.0, 1092.0, 1078.0],
    [1616.0, 0.0, 1918.0, 1072.0],
    [423.0, 469.0, 474.0, 516.0],
    [1160.0, 736.0, 1258.0, 842.0],
    [442.0, 589.0, 561.0, 656.0],
    [770.0, 1046.0, 832.0, 1080.0],
    [766.0, 1046.0, 870.0, 1080.0],
    [441.0, 491.0, 498.0, 604.0],
    [514.0, 462.0, 540.0, 510.0],
    [87.0, 830.0, 1100.0, 1079.0],
    [270.0, 214.0, 436.0, 275.0],
    [899.0, 317.0, 967.0, 400.0],
    [430.0, 549.0, 564.0, 657.0],
    [896.0, 299.0, 1001.0, 402.0],
    [1592.0, 813.0, 1814.0, 1079.0],
    [1190.0, 232.0, 1264.0, 365.0],
    [109.0, 374.0, 260.0, 671.0],
    [432.0, 479.0, 558.0, 601.0],
    [497.0, 554.0, 554.0, 645.0],
    [423.0, 590.0, 473.0, 645.0],
    [1271.0, 699.0, 1367.0, 728.0],
    [1515.0, 835.0, 1546.0, 1078.0],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call vlx-seek-1.5-3b-sanitized through raw vLLM."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("VLLM_CHAT_URL", DEFAULT_URL),
        help="OpenAI-compatible chat-completions endpoint.",
    )
    parser.add_argument(
        "--model", default=os.getenv("MODEL_NAME", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_QUESTION,
    )
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        help=(
            "Image URL or local path. Repeat for multiple images; the first "
            "explicit value replaces the built-in default image."
        ),
    )
    bbox_group = parser.add_mutually_exclusive_group()
    bbox_group.add_argument(
        "--bbox-json",
        help='Bounding boxes as JSON, for example "[[0,0,100,100]]".',
    )
    bbox_group.add_argument(
        "--bbox-file", type=Path, help="Path to a JSON file containing boxes."
    )
    parser.add_argument("--max-tokens", type=int, default=27)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("VLLM_REQUEST_TIMEOUT", "600")),
    )
    parser.add_argument(
        "--api-key", default=os.getenv("APIKEY"), help="Optional vLLM API key."
    )
    parser.add_argument(
        "--raw", action="store_true", help="Print the complete JSON response."
    )
    args = parser.parse_args()
    if args.image is None:
        args.image = [DEFAULT_IMAGE_URL]
    return args


def local_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme in {"http", "https", "data"}:
        return None
    return Path(value).expanduser()


def image_source(value: str) -> str:
    path = local_path(value)
    if path is None:
        return value
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Image file not found: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_bboxes(args: argparse.Namespace) -> Any | None:
    if args.bbox_json:
        value = json.loads(args.bbox_json)
    elif args.bbox_file:
        value = json.loads(args.bbox_file.read_text(encoding="utf-8"))
    else:
        return DEFAULT_BBOX_LIST
    if not isinstance(value, list):
        raise ValueError("Bounding boxes must be a JSON list.")
    return value


def build_payload(args: argparse.Namespace, bboxes: Any | None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_source(image)}}
        for image in args.image
    ]
    content.append({"type": "text", "text": args.prompt})

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": args.stream,
        "chat_template_kwargs": {"disable_thinking_tags": True},
        "skip_special_tokens": False,
        "stop": ["<|im_end|>"],
        "stop_token_ids": [248046],
        "include_stop_str_in_output": False,
    }
    if bboxes is not None:
        if not args.image:
            raise ValueError("--bbox-json/--bbox-file requires at least one --image.")
        payload["mm_processor_kwargs"] = {"bbox_list": bboxes}
    return payload


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def print_response(data: dict[str, Any], raw: bool) -> None:
    if raw:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    choice = data["choices"][0]
    message = choice.get("message", {})
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    print(f"{reasoning}{content}")
    if choice.get("result_bbox_list"):
        print("\nresult_bbox_list:")
        print(json.dumps(choice["result_bbox_list"], ensure_ascii=False, indent=2))
    if data.get("usage"):
        print("\nusage:", json.dumps(data["usage"], ensure_ascii=False))


def stream_response(response: requests.Response, raw: bool) -> None:
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        data = json.loads(body)
        if raw:
            print(json.dumps(data, ensure_ascii=False))
            continue
        delta = data.get("choices", [{}])[0].get("delta", {})
        text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
        if text:
            print(text, end="", flush=True)
    if not raw:
        print()


def main() -> int:
    args = parse_args()
    try:
        payload = build_payload(args, load_bboxes(args))
        with requests.post(
            args.url,
            headers=request_headers(args.api_key),
            json=payload,
            stream=args.stream,
            timeout=args.timeout,
        ) as response:
            if not response.ok:
                print(
                    f"HTTP {response.status_code}: {response.text}",
                    file=sys.stderr,
                )
                return 1
            if args.stream:
                stream_response(response, args.raw)
            else:
                print_response(response.json(), args.raw)
    except (OSError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
