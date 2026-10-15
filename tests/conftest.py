"""테스트 공통 설정"""
import pytest


@pytest.fixture(autouse=True)
def _no_sample_seeding(monkeypatch):
    """API 테스트의 임시 uploads 폴더에 저장소 샘플(영상 10 MB)이 복사되지 않도록"""
    monkeypatch.setenv("SOCCER_SEED_SAMPLES", "0")
