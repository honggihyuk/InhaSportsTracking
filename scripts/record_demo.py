"""
데모 영상 녹화 — 앱을 Playwright 헤드리스 Chrome(1920×1080)으로 조작하며 화면을 녹화하고 H.264 MP4 로 변환

  python scripts/record_demo.py --url http://localhost:5173/ --out D:/TermProject/recordings \
      --question "스프린트 장면을 보여줘"

장면
  A. 메인 화면: 중계 영상 재생 + 검출 박스(팀 색, GK/REF, '#트랙ID · 등번호'), 경기장 라인 겹쳐 보기
  B. 3D 디지털 트윈: 자유 시점에서 천천히 돌려 보고, 탑다운·선수 시점으로 바꿔 가며 리플레이
  C. 경기 지표 패널 (팀 ↔ 선수)
  D. 경기 질의: 질문 입력 → 답변과 근거 장면 이동 (답변을 기다리는 구간은 잘라 냄)

헤드리스 녹화에는 마우스 커서가 찍히지 않으므로 페이지에 가짜 커서를 그린다. 설치된 Chrome 을 쓴다
(pip install playwright, 브라우저 다운로드 불필요). ffmpeg 는 imageio-ffmpeg 의 실행 파일을 쓴다.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

CURSOR_JS = """
(() => {
  const add = () => {
    if (document.getElementById('__demo_cursor')) return;
    const c = document.createElement('div');
    c.id = '__demo_cursor';
    c.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;z-index:2147483647;pointer-events:none;' +
      'border:2px solid #fff;border-radius:50%;background:rgba(215,255,60,.55);box-shadow:0 0 6px rgba(0,0,0,.6);' +
      'transform:translate(-50%,-50%);transition:transform .08s';
    document.body.appendChild(c);
    addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    addEventListener('mousedown', () => c.style.transform = 'translate(-50%,-50%) scale(.7)', true);
    addEventListener('mouseup', () => c.style.transform = 'translate(-50%,-50%)', true);
  };
  if (document.body) add(); else addEventListener('DOMContentLoaded', add);
})();
"""


class Demo:
    def __init__(self, page, t0):
        self.page, self.t0 = page, t0
        self.marks = []
        self.x, self.y = 960, 540

    def now(self):
        return round(time.monotonic() - self.t0, 2)

    def mark(self, name):
        self.marks.append((name, self.now()))
        print(f'[{self.now():6.1f}s] {name}')

    def pause(self, s=1.5):
        self.page.wait_for_timeout(int(s * 1000))

    def move(self, x, y, steps=25):
        self.page.mouse.move(x, y, steps=steps)
        self.x, self.y = x, y

    def move_to(self, locator, steps=25):
        box = locator.bounding_box()
        self.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2, steps)
        return box

    def click(self, locator, wait=1.2):
        locator.scroll_into_view_if_needed()
        self.move_to(locator)
        self.pause(0.3)
        locator.click()
        self.pause(wait)

    def drag(self, x0, y0, x1, y1, steps=60):
        self.move(x0, y0, 15)
        self.page.mouse.down()
        self.page.mouse.move(x1, y1, steps=steps)
        self.page.mouse.up()
        self.x, self.y = x1, y1


def run(url, out_dir: Path, question: str, answer_timeout: float):
    from playwright.sync_api import sync_playwright
    raw = out_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True,
                                    args=['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist',
                                          '--autoplay-policy=no-user-gesture-required'])
        ctx = browser.new_context(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1,
                                  record_video_dir=str(raw), record_video_size={'width': 1920, 'height': 1080})
        ctx.add_init_script(CURSOR_JS)
        t0 = time.monotonic()
        page = ctx.new_page()
        d = Demo(page, t0)
        page.goto(url)
        page.get_by_text('PLAY', exact=True).wait_for(timeout=60000)
        page.wait_for_timeout(3000)                      # 첫 화면·영상 로드
        video = page.locator('video').first

        # A. 메인 화면 -------------------------------------------------------
        d.mark('A_main')
        d.move_to(video)
        d.pause(1.5)
        d.click(page.get_by_text('PLAY', exact=True), wait=0.5)
        vb = video.bounding_box()
        for fx, fy in ((0.3, 0.45), (0.55, 0.5), (0.7, 0.4), (0.45, 0.6)):   # 박스·라벨을 따라 천천히
            d.move(vb['x'] + vb['width'] * fx, vb['y'] + vb['height'] * fy, steps=40)
            d.pause(1.2)
        lines = page.get_by_text('경기장 라인', exact=True)
        d.click(lines, wait=4.0)                          # 경기장 라인 겹쳐 보기
        d.click(lines, wait=1.0)

        # B. 3D 디지털 트윈 ----------------------------------------------------
        d.mark('B_twin')
        d.click(page.get_by_role('button', name='처음으로'), wait=0.6)     # 처음부터 다시 재생하며 3D 로
        if page.get_by_role('button', name='재생').count():
            d.click(page.get_by_role('button', name='재생'), wait=0.4)
        d.click(page.get_by_text('3D', exact=True), wait=2.0)
        boxes = [c.bounding_box() for c in page.locator('canvas:visible').all()]
        cb = max((b for b in boxes if b), key=lambda b: b['width'] * b['height'])    # 3D 트윈 WebGL 캔버스
        cx, cy = cb['x'] + cb['width'] / 2, cb['y'] + cb['height'] / 2
        d.click(page.get_by_text('자유 시점', exact=True), wait=1.0)
        d.drag(cx - 250, cy, cx + 250, cy - 60, steps=120)       # 천천히 돌려 보기
        d.pause(1.5)
        d.move(cx, cy, 20)
        for _ in range(4):                                       # 조금 다가가기
            page.mouse.wheel(0, -240)
            d.pause(0.35)
        d.drag(cx + 200, cy - 40, cx - 150, cy + 40, steps=120)
        d.pause(2.0)
        d.mark('B_topdown')
        d.click(page.get_by_text('탑다운', exact=True), wait=4.0)
        d.mark('B_player_view')
        d.click(page.get_by_text('선수 시점', exact=True), wait=5.0)
        d.click(page.get_by_text('자유 시점', exact=True), wait=1.5)
        d.click(page.get_by_text('분할', exact=True), wait=2.5)
        pause_btn = page.get_by_role('button', name='일시정지')
        if pause_btn.count():
            d.click(pause_btn, wait=0.8)                         # 재생 멈춤

        # C. 경기 지표 ------------------------------------------------------
        d.mark('C_stats')
        page.mouse.wheel(0, 400)
        d.pause(1.5)
        stats_card = page.get_by_text('경기 지표', exact=True)
        d.move_to(stats_card)
        d.pause(1.5)
        d.click(page.get_by_text('선수', exact=True).last, wait=3.0)
        d.click(page.get_by_text('팀', exact=True).last, wait=2.0)

        # D. 경기 질의 ------------------------------------------------------
        d.mark('D_ask')
        box = page.get_by_placeholder('경기에 대해 물어보세요')
        d.click(box, wait=0.5)
        page.keyboard.type(question, delay=90)
        d.pause(0.8)
        n_answers = page.locator('.ask-log .ask-q').count()
        page.keyboard.press('Enter')
        d.mark('D_asked')
        busy = page.get_by_text('분석 데이터를 살펴보는 중', exact=False)
        deadline = time.monotonic() + answer_timeout
        answered = False
        page.wait_for_timeout(800)
        while time.monotonic() < deadline:
            if busy.count() == 0 and page.locator('.ask-log .ask-q').count() > n_answers:
                answered = True
                break
            page.wait_for_timeout(500)
        d.mark('D_answer' if answered else 'D_timeout')
        last = page.locator('.ask-log > li').last
        if last.count():
            d.move_to(last, steps=30)
        d.pause(5.0)
        evidence = page.locator('.ask-log .ask-frames button')     # 답변의 근거 시점 버튼
        if evidence.count():
            d.move_to(evidence.first, steps=30)
            d.pause(1.0)
            d.click(evidence.first, wait=1.5)
            d.mark('D_evidence')
            page.mouse.wheel(0, -400)
            d.pause(1.5)
            d.move_to(page.locator('video').first, steps=30)
            if page.get_by_role('button', name='재생').count():
                d.click(page.get_by_role('button', name='재생'), wait=5.0)   # 근거 장면부터 잠깐 재생
            else:
                d.pause(5.0)
        d.pause(2.0)
        d.mark('end')
        page_video = page.video
        ctx.close()
        browser.close()
        webm = Path(page_video.path())
    return webm, d.marks


def ffmpeg_exe():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(webm: Path, marks, out_dir: Path, name: str, max_wait: float = 4.0):
    """webm → H.264 MP4. 답변 대기 구간이 max_wait 보다 길면 잘라 내고, 장면별 클립도 만든다"""
    ff = ffmpeg_exe()
    m = dict(marks)
    keep = [(0.0, m['end'] + 1.0)]
    cut = None
    if 'D_asked' in m and 'D_answer' in m and m['D_answer'] - m['D_asked'] > max_wait + 1.0:
        a, b = m['D_asked'] + max_wait / 2, m['D_answer'] - max_wait / 2
        keep = [(0.0, a), (b, m['end'] + 1.0)]
        cut = (a, b)
    parts = ''.join(f'[0:v]trim={s}:{e},setpts=PTS-STARTPTS[v{i}];' for i, (s, e) in enumerate(keep))
    concat = ''.join(f'[v{i}]' for i in range(len(keep))) + f'concat=n={len(keep)}:v=1:a=0[out]'
    out = out_dir / f'{name}.mp4'
    subprocess.run([ff, '-y', '-i', str(webm), '-filter_complex', parts + concat, '-map', '[out]', '-r', '30',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(out)], check=True, capture_output=True)

    def shifted(t):                                   # 원본 시각 → 잘라 낸 뒤 시각
        if cut and t >= cut[1]:
            return t - (cut[1] - cut[0])
        return t
    scenes = [(k, shifted(v)) for k, v in marks if not k.startswith('D_asked')]
    starts = [(k, t) for k, t in scenes if k in ('A_main', 'B_twin', 'C_stats', 'D_ask')]
    clips = []
    total = shifted(m['end'] + 1.0)
    for i, (k, t) in enumerate(starts):
        e = starts[i + 1][1] if i + 1 < len(starts) else total
        clip = out_dir / f'{name}_{k}.mp4'
        subprocess.run([ff, '-y', '-ss', f'{t:.2f}', '-to', f'{e:.2f}', '-i', str(out), '-c:v', 'libx264', '-preset', 'medium',
                        '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(clip)], check=True, capture_output=True)
        clips.append(clip.name)
    return out, scenes, cut, clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:5173/')
    ap.add_argument('--out', type=Path, default=Path('D:/TermProject/recordings'))
    ap.add_argument('--name', default='demo_sports_video_1')
    ap.add_argument('--question', default='스프린트 장면을 보여줘')
    ap.add_argument('--answer-timeout', type=float, default=240.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    webm, marks = run(a.url, a.out, a.question, a.answer_timeout)
    mp4, scenes, cut, clips = encode(webm, marks, a.out, a.name)
    info = {'video': str(mp4), 'raw': str(webm), 'scenes': scenes, 'cut_wait': cut, 'clips': clips,
            'question': a.question}
    (a.out / f'{a.name}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
