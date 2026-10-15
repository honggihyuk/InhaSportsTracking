"""
에이전트용 채팅 모델 (3부 22.2~22.3절)

- QwenVLChat: 로컬 Qwen3-VL (transformers). 도구 정의를 채팅 템플릿에 넣고, 모델이 낸
  <tool_call>{...}</tool_call> 을 직접 파싱한다 (LangChain ChatHuggingFace 는 로컬 파이프라인의
  tool call 을 파싱하지 못하는 알려진 문제가 있어 쓰지 않음)
- OpenAIChat: OpenAI 호환 서버 (vLLM·Ollama 로컬 서버, 또는 유료 API) — base_url 과 모델명만 바꾸면 된다

두 클래스 모두 chat(messages, tools) -> {'content': str, 'tool_calls': [{'name', 'arguments'}]}.
messages 는 OpenAI 형식에 가깝게: {'role', 'content'} 이며 content 는 문자열 또는
[{'type': 'text', 'text'}, {'type': 'image', 'image': BGR ndarray}] 목록.
"""

import base64
import json
import re
from typing import Dict, List, Optional

import numpy as np

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text: str):
    """Qwen 계열 출력에서 tool call 추출 → (남은 본문, [{'name', 'arguments'}])"""
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get('arguments') or obj.get('parameters') or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if obj.get('name'):
            calls.append({'name': obj['name'], 'arguments': args})
    content = TOOL_CALL_RE.sub('', text)
    content = re.sub(r"<think>.*?</think>", '', content, flags=re.S).strip()
    return content, calls


def _pil(img_bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))


class QwenVLChat:
    """로컬 Qwen3-VL (2B/4B/8B Instruct). 모델은 첫 질의 때 GPU 에 올린다"""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cuda",
                 max_new_tokens: int = 768, max_pixels: int = 960 * 540):
        self.model_id, self.device, self.max_new_tokens, self.max_pixels = model_id, device, max_new_tokens, max_pixels
        self.model = self.processor = None

    def _load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dtype = torch.bfloat16 if str(self.device).startswith('cuda') else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, max_pixels=self.max_pixels)

    @staticmethod
    def _to_hf(messages: List[Dict]) -> List[Dict]:
        out = []
        for m in messages:
            content = m['content']
            if isinstance(content, list):
                content = [{'type': 'image', 'image': _pil(c['image'])} if c['type'] == 'image' else c for c in content]
            msg = {'role': m['role'], 'content': content}
            if m.get('tool_calls'):  # 어시스턴트의 도구 호출 기록은 템플릿 형식으로
                msg['tool_calls'] = [{'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}}
                                     for c in m['tool_calls']]
            out.append(msg)
        return out

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        import torch
        if self.model is None:
            self._load()
        inputs = self.processor.apply_chat_template(
            self._to_hf(messages), tools=tools, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        content, calls = parse_tool_calls(text)
        return {'content': content, 'tool_calls': calls}

    def unload(self):
        self.model = self.processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


class OpenAIChat:
    """OpenAI 호환 /chat/completions (vLLM, Ollama, LM Studio, 유료 API)"""

    def __init__(self, base_url: str, model: str, api_key: str = 'local', max_new_tokens: int = 768,
                 timeout: float = 120.0):
        self.base_url, self.model, self.api_key = base_url.rstrip('/'), model, api_key
        self.max_new_tokens, self.timeout = max_new_tokens, timeout

    @staticmethod
    def _to_openai(messages: List[Dict]) -> List[Dict]:
        import cv2
        out = []
        for i, m in enumerate(messages):
            content = m['content']
            if isinstance(content, list):
                parts = []
                for c in content:
                    if c['type'] == 'image':
                        ok, buf = cv2.imencode('.jpg', c['image'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                        url = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
                        parts.append({'type': 'image_url', 'image_url': {'url': url}})
                    else:
                        parts.append(c)
                content = parts
            msg = {'role': m['role'], 'content': content}
            if m.get('tool_calls'):
                msg['tool_calls'] = [{'id': f'call_{i}_{k}', 'type': 'function',
                                      'function': {'name': c['name'], 'arguments': json.dumps(c['arguments'])}}
                                     for k, c in enumerate(m['tool_calls'])]
            if m['role'] == 'tool':
                msg['tool_call_id'] = m.get('tool_call_id', f'call_{i}')
            out.append(msg)
        return out

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        import urllib.request
        body = {'model': self.model, 'messages': self._to_openai(messages), 'max_tokens': self.max_new_tokens,
                'temperature': 0}
        if tools:
            body['tools'] = tools
        req = urllib.request.Request(f'{self.base_url}/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Authorization': f'Bearer {self.api_key}',
                                              'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            msg = json.loads(resp.read())['choices'][0]['message']
        calls = [{'name': c['function']['name'], 'arguments': json.loads(c['function'].get('arguments') or '{}')}
                 for c in msg.get('tool_calls') or []]
        content, inline = parse_tool_calls(msg.get('content') or '')  # 일부 서버는 본문에 tool call 을 남김
        return {'content': content, 'tool_calls': calls or inline}

    def unload(self):
        pass
