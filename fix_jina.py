r"""
jina-embeddings-v3 원격 코드 캐시 복구 스크립트.

증상:
  FileNotFoundError: ...transformers_modules\jinaai\xlm_..._implementation\<sha>\block.py
  (재실행할 때마다 없는 파일 이름만 바뀜)

원인:
  transformers가 상대 임포트를 따라가며 파일을 한 개씩 받는데,
  Windows 심볼릭 링크 미지원 등으로 부분 다운로드 상태에 빠짐.

해결:
  jinaai/xlm-roberta-flash-implementation 저장소의 .py 파일 전체를
  캐시 모듈 디렉터리에 한 번에 복사해 넣습니다.

사용:
  python fix_jina.py
"""
import shutil
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download

MODULES_ROOT = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
IMPL_REPO = "jinaai/xlm-roberta-flash-implementation"


def try_load():
    from encoders import Encoder
    Encoder("jina-v3").free()


def find_module_dirs() -> list[Path]:
    """캐시에 만들어진 xlm-roberta-flash-implementation 모듈 디렉터리들을 찾습니다.

    transformers가 하이픈을 '_hyphen_'으로 바꿔 저장하므로 이름이 특이합니다.
    """
    base = MODULES_ROOT / "jinaai"
    if not base.exists():
        return []
    out = []
    for d in base.iterdir():
        name = d.name.lower()
        if "xlm" in name and "flash" in name and "implementation" in name:
            # 그 아래 커밋 해시 폴더들
            out.extend([c for c in d.iterdir() if c.is_dir() and c.name != "__pycache__"])
    return out


def main():
    # 1) 모듈 디렉터리를 만들기 위해 일단 한 번 로드 시도 (실패해도 됨)
    if not find_module_dirs():
        print("[1/3] 캐시 디렉터리 생성을 위해 로드를 한 번 시도합니다...")
        try:
            try_load()
            print("성공했습니다. 복구가 필요 없습니다.")
            return
        except Exception as e:
            print(f"  예상된 실패: {type(e).__name__}")

    dirs = find_module_dirs()
    if not dirs:
        print("모듈 디렉터리를 찾지 못했습니다. 네트워크나 HF 접근을 확인하세요.")
        return
    print(f"[2/3] 대상 디렉터리 {len(dirs)}개: {[d.parent.name + '/' + d.name[:8] for d in dirs]}")

    # 2) 각 커밋 해시에 맞춰 저장소 전체를 받아 .py 를 복사
    for d in dirs:
        revision = d.name  # 폴더명 = 커밋 해시
        try:
            src = Path(snapshot_download(IMPL_REPO, revision=revision,
                                         allow_patterns=["*.py", "*.json"]))
        except Exception:
            print(f"  revision={revision[:8]} 다운로드 실패 -> main으로 재시도")
            src = Path(snapshot_download(IMPL_REPO, allow_patterns=["*.py", "*.json"]))

        n = 0
        for f in src.glob("*.py"):
            shutil.copy2(f, d / f.name)
            n += 1
        print(f"  {d.name[:8]} <- {n}개 파일 복사")

    # 3) 재시도
    print("[3/3] 다시 로드합니다...")
    try:
        try_load()
        print("\n복구 성공. 이제 아래를 실행하세요:")
        print("  python encoders.py jina-v3")
    except Exception:
        traceback.print_exc()
        print("\n여전히 실패합니다. config.py의 DEFAULT_MODELS에서 'jina-v3'를 제외하고")
        print("나머지 5개 모델로 진행하는 것을 권합니다.")


if __name__ == "__main__":
    main()