"""
경기 질의 에이전트 (3부 22.1절) — 질문 → (도구 호출 ↔ 결과) 반복 → 답변 + 근거 프레임

LangGraph 의 ReAct 루프와 같은 일을 하되, 로컬 Qwen3-VL 의 tool call 을 직접 다루기 위해 직접 구현했다.
도구가 이미지를 돌려주면(get_frame, crop_player) 다음 사용자 메시지로 이미지를 붙여 VLM 이 보게 한다.
"""

import json
from typing import Dict, List, Optional

from .tools import MatchTools

SYSTEM_PROMPT = """너는 축구 경기 영상 분석 도우미다. 사용자의 질문에 한국어로 답한다.
규칙:
1. 수치(거리·속도·점유율·좌표·횟수)는 반드시 도구 결과에서만 가져온다. 추측해서 숫자를 만들지 않는다.
2. 필요한 도구를 먼저 호출하고, 결과를 받은 뒤 답한다. 한 번에 여러 도구를 호출해도 된다.
   사용자가 말하지 않은 조건(팀, 구간, 선수)은 도구 인자에 넣지 않는다. 전체를 물으면 전체를 조회한다.
3. 답변에는 근거가 된 시점을 '12.3초' 처럼 적는다.
4. 선수는 트랙 ID 로 부른다 (예: #7). 등번호나 이름은 이미지에서 확인한 경우에만 말한다.
5. 장면을 직접 봐야 판단할 수 있는 질문(누가 공을 가졌는지, 어떤 상황인지)만 get_frame/crop_player 로 이미지를 본다.
6. 도구가 오류나 데이터 없음을 알리면 그 사실을 그대로 전하고, 무엇이 필요한지 말한다.
7. 답변은 간결하게, 핵심부터."""


NUDGE = ('방금 답은 도구로 확인하지 않았다. 경기 데이터에 관한 질문이면 필요한 도구를 호출해서 확인한 뒤 다시 답해라. '
         '데이터가 필요 없는 질문(인사 등)이면 같은 답을 그대로 다시 말해라.')


def _summary(result: Dict) -> str:
    """도구 결과를 모델에 넘길 텍스트로 (이미지·근거 프레임 목록은 빼고)"""
    data = {k: v for k, v in result.items() if k not in ('image', 'frames')}
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= 6000 else text[:6000] + ' …(생략)'


class MatchAgent:
    def __init__(self, llm, tools: MatchTools, max_steps: int = 6, system_prompt: str = SYSTEM_PROMPT):
        self.llm, self.tools, self.max_steps, self.system_prompt = llm, tools, max_steps, system_prompt

    def ask(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        """
        Returns:
            {'answer': str, 'steps': [{'tool', 'arguments', 'result'}], 'frames': [근거 프레임, 중복 제거]}
        """
        overview = _summary(self.tools.match_overview())
        messages: List[Dict] = [{'role': 'system', 'content': f"{self.system_prompt}\n\n경기 개요: {overview}"}]
        messages += history or []
        messages.append({'role': 'user', 'content': question})
        steps, frames = [], []
        schemas = self.tools.schemas()
        nudged = False
        for _ in range(self.max_steps):
            reply = self.llm.chat(messages, schemas)
            calls = reply.get('tool_calls') or []
            if not calls and not steps and not nudged:
                # 작은 로컬 모델은 이전 대화만 보고 수치를 지어내기도 한다 → 한 번은 도구로 확인하게 한다
                nudged = True
                messages.append({'role': 'assistant', 'content': reply.get('content') or ''})
                messages.append({'role': 'user', 'content': NUDGE})
                continue
            if not calls:
                return {'answer': reply.get('content') or '', 'steps': steps, 'frames': _unique(frames)}
            messages.append({'role': 'assistant', 'content': reply.get('content') or '', 'tool_calls': calls})
            images = []
            for call in calls:
                result = self.tools.call(call['name'], call.get('arguments') or {})
                frames += [int(f) for f in result.get('frames') or []]
                steps.append({'tool': call['name'], 'arguments': call.get('arguments') or {},
                              'result': {k: v for k, v in result.items() if k != 'image'}})
                messages.append({'role': 'tool', 'name': call['name'], 'content': _summary(result)})
                if result.get('image') is not None:
                    images.append((call['name'], result))
            if images:  # 이미지 결과는 사용자 메시지로 붙여야 대부분의 채팅 템플릿이 받아준다
                content = []
                for name, res in images:
                    content.append({'type': 'text', 'text': f"[{name} 결과 이미지: {res.get('time_s')}초]"})
                    content.append({'type': 'image', 'image': res['image']})
                messages.append({'role': 'user', 'content': content})
        # 도구 호출만 반복하면 마지막으로 도구 없이 답하게 한다
        messages.append({'role': 'user', 'content': '지금까지의 도구 결과만으로 답해줘.'})
        reply = self.llm.chat(messages, None)
        return {'answer': reply.get('content') or '', 'steps': steps, 'frames': _unique(frames)}


def _unique(frames: List[int]) -> List[int]:
    seen, out = set(), []
    for f in frames:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out[:10]


def make_llm(config: Dict, device: str = 'cuda'):
    """agent 설정 → 채팅 모델. provider: local (Qwen3-VL, transformers) | openai (OpenAI 호환 서버)"""
    from .llm import OpenAIChat, QwenVLChat
    cfg = config or {}
    if cfg.get('provider', 'local') == 'openai':
        import os
        return OpenAIChat(cfg.get('base_url', 'http://localhost:8001/v1'), cfg.get('model', 'Qwen/Qwen3-VL-4B-Instruct'),
                          api_key=os.environ.get(cfg.get('api_key_env', 'OPENAI_API_KEY'), 'local'),
                          max_new_tokens=cfg.get('max_new_tokens', 768))
    return QwenVLChat(cfg.get('model', 'Qwen/Qwen3-VL-4B-Instruct'), device=device,
                      max_new_tokens=cfg.get('max_new_tokens', 768))
