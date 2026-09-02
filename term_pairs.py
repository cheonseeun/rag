# -*- coding: utf-8 -*-
"""
한국 회계기준 도메인의 용어 쌍 목록.

confusable_pairs: 표면적으로 비슷해 보이지만 회계처리가 명확히 다른 개념 쌍.
                   파인튜닝 후 유사도가 "낮아지길" 기대하는 그룹.
antonym_pairs:     정반대 의미인데 한두 글자만 다른 금융 거래 용어 쌍 (매수/매도 등).
                   confusable보다 더 위험한 케이스 -- 헷갈리면 결론이 완전히 반대가 됨.
                   파인튜닝 후 유사도가 "확실히 낮아지길" 기대.
synonym_pairs:     실질적으로 같은 개념을 가리키는(또는 거의 같은) 용어 쌍.
                   파인튜닝 후에도 유사도가 "유지되길" 기대하는 대조군.
                   이게 같이 떨어지면 모델이 무차별적으로 다 밀어낸 것이라는 경고 신호.
unrelated_pairs:   서로 무관한 개념. 항상 낮은 유사도를 유지해야 하는 기준선.

주의: 이 목록은 도메인 지식 기반으로 초안을 잡은 것입니다. 실제 회계 처리
기준으로 세은님이 검토·보완하시는 걸 권합니다. 특히 mine_hard_negatives.py가
실제로 뽑은 negative 쌍에서 추가 후보를 뽑는 함수도 아래 제공합니다
(도메인 지식 없이도 데이터 기반으로 검증 가능).
"""

CONFUSABLE_PAIRS = [
    ("대손충당금", "대손준비금"),
    ("이익잉여금", "자본잉여금"),
    ("자기주식", "자본금"),
    ("유동부채", "비유동부채"),
    ("매출채권", "매입채무"),
    ("회계정책", "회계추정치"),
    ("연결재무제표", "별도재무제표"),
    ("확정급여채무", "확정기여채무"),
    ("지분법", "원가법"),
    ("금융자산", "금융부채"),
    ("감가상각비", "감모상각비"),
    ("재평가잉여금", "재평가손익"),
    ("자산손상", "자산재평가"),
    ("전기오류수정", "회계추정치변경"),
    ("영업권", "무형자산"),
    ("우발부채", "충당부채"),
    ("리스부채", "리스자산"),
    ("당기순이익", "포괄손익"),
]

# 금융 거래 반의어: 표면적 유사성(같은 자모/음절 구조)은 높지만 의미가 정반대.
# 회계기준 조문이 아니라 실거래/시황 문맥에서 흔히 쓰이는 용어들이라,
# 짧은 단어 쌍뿐 아니라 실제 문장 맥락에서 재는 문장형도 같이 넣었습니다.
ANTONYM_PAIRS = [
    ("매수", "매도"),
    ("매수 주문", "매도 주문"),
    ("매수 포지션", "매도 포지션"),
    ("공매도", "공매수"),
    ("차입", "대여"),
    ("차입금", "대여금"),
    ("매출", "매입"),
    ("입금", "출금"),
    ("상승세", "하락세"),
    ("강세장", "약세장"),
    ("자산 증가", "자산 감소"),
    ("이익 실현", "손실 실현"),
    ("배당 지급", "배당 수취"),
    ("채권자", "채무자"),
    ("대주주", "소액주주"),
    ("장기 투자", "단기 투자"),
    ("이 종목은 지금 매수하기 좋은 시점이다", "이 종목은 지금 매도하기 좋은 시점이다"),
    ("금리가 상승하면 채권 가격은 하락한다", "금리가 하락하면 채권 가격은 상승한다"),
]

SYNONYM_PAIRS = [
    ("대손충당금", "대손충당금 설정액"),
    ("당기순이익", "당기순손익"),
    ("유형자산", "유형자산의 취득원가"),
    ("재무상태표", "대차대조표"),
    ("자기주식", "자사주"),
    ("연결재무제표", "연결 재무제표"),
    ("감가상각비", "감가상각액"),
    ("충당부채", "충당금"),
    ("포괄손익계산서", "손익계산서"),
    ("금융상품", "금융자산 및 금융부채"),
    # 매수/매도 대조군: antonym이 아니라 진짜 같은 개념(동의어)인 쌍
    ("매수", "매입"),
    ("매도", "매각"),
    ("공매도", "숏 포지션"),
]

UNRELATED_PAIRS = [
    ("대손충당금", "직원 연차수당"),
    ("이익잉여금", "사내 회의실 예약"),
    ("연결재무제표", "점심 메뉴 추천"),
    ("자기주식", "오늘의 날씨"),
    ("금융자산", "축구 경기 결과"),
    ("매수", "오늘의 날씨"),
]

# ---------------------------------------------------------------------------
# 수작업 큐레이션 회계 도메인 반의어/유의어 50쌍
# ---------------------------------------------------------------------------
# 공개된 한국어 회계 도메인 유의어/반의어 전용 데이터셋이 없어(용어집은 있지만
# "관계 라벨"이 없음) 도메인 지식으로 직접 구성했습니다. ACCOUNTING_ANTONYM_PAIRS,
# SYNONYM_PAIRS와 중복되지 않게 새로 작성했습니다.
# 주의: triplet 데이터에서 실제 등장을 확인한 자동 추출본이 아니라 지식 기반 초안이므로
# 반드시 세은님의 회계 지식으로 검토 후 사용하세요. 특히 "고정자산/비유동자산"처럼
# 신구 용어 대응인 경우 최신 K-IFRS 기준 표현인지 재확인이 필요합니다.

ACCOUNTING_ANTONYM_PAIRS_CURATED = [
    ("유동자산", "유동부채"),
    ("차입", "대여"),
    ("원가 상승", "원가 하락"),
    ("부채비율 증가", "부채비율 감소"),
    ("이익잉여금 증가", "이익잉여금 감소"),
    ("재고자산 증가", "재고자산 감소"),
    ("매출채권 증가", "매출채권 감소"),
    ("대손상각비 증가", "대손상각비 감소"),
    ("유형자산 취득", "유형자산 처분"),
    ("무형자산 취득", "무형자산 처분"),
    ("자기주식 취득", "자기주식 처분"),
    ("배당금 지급", "배당금 수령"),
    ("이자 지급", "이자 수취"),
    ("대변잔액", "차변잔액"),
    ("자산재평가 증가", "자산재평가 감소"),
    ("순매입액", "순매출액"),
    ("당기순이익", "당기순손실"),
    ("영업이익", "영업손실"),
    ("매출총이익", "매출총손실"),
    ("현금흐름 유입", "현금흐름 유출"),
    ("부채 상환", "부채 발생"),
    ("재무상태 개선", "재무상태 악화"),
    ("자산 감액", "자산 증액"),
    ("수익적 지출", "자본적 지출"),  # 성격이 다른 지출 분류(반의어보다 대조 개념에 가까움 -- 검수 시 재확인 권장
    ("평가이익", "평가손실"),
]

ACCOUNTING_SYNONYM_PAIRS_CURATED = [
    ("재고자산", "재고"),
    ("매출원가", "판매원가"),
    ("유동부채", "단기부채"),
    ("비유동부채", "장기부채"),
    ("유동자산", "단기자산"),
    ("비유동자산", "고정자산"),  # 구용어(고정자산)-신용어(비유동자산) 대응, K-IFRS 전환 이력 확인 권장
    ("외상매출금", "매출채권"),  # 구용어-신용어 대응
    ("외상매입금", "매입채무"),  # 구용어-신용어 대응
    ("자본금", "납입자본"),
    ("이익잉여금", "유보이익"),
    ("원가법", "취득원가법"),
    ("정액법", "균등상각법"),
    ("정률법", "체감잔액법"),
    ("순이익", "당기순이익"),
    ("매출액", "매출"),
    ("영업활동현금흐름", "영업활동으로 인한 현금흐름"),
    ("재무제표주석", "주석"),
    ("감가상각누계액", "감가상각충당금"),  # 구용어 대응, 현행 기준 표현 재확인 권장
    ("차입금", "부채성 자금"),  # 다소 넓은 대응, 검수 시 재확인 권장
    ("자본잉여금", "주식발행초과금 등 잉여금"),  # 상위개념-대표사례 관계에 가까움, 검수 필요
    ("법인세비용", "법인세"),
    ("감가상각비", "상각비"),
    ("무형자산상각비", "무형자산 상각액"),
    ("퇴직급여충당부채", "퇴직급여충당금"),  # 구용어-신용어 대응
    ("대손충당금", "대손충당금 전입액"),
]

# 회계 도메인 내부의 진짜 반의어 쌍: ANTONYM_PAIRS(실거래/시황 용어)와 대조하기 위한 그룹.
# 학습 데이터(AI-Hub 기업 회계처리 기준 데이터)와 같은 도메인 안에서 반의어 구별력을
# 측정하기 위해 별도로 분리했습니다. confusable과 비교하면:
#   - 이것도 confusable만큼 잘 떨어진다면 -> ANTONYM_PAIRS의 "개선폭 작음"은
#     순전히 도메인 갭(실거래 용어를 학습 안 해서) 때문
#   - 이것도 안 떨어진다면 -> "표면적 유사성(confusable)"과 "의미상 반대(antonym)"라는
#     관계 종류 자체를 모델이 다르게 취급한다는 더 강한 근거
#     (하드 네거티브 마이닝이 confusable 스타일 negative만 학습시켰을 가능성)
ACCOUNTING_ANTONYM_PAIRS = [
    ("차변", "대변"),
    ("자산", "부채"),
    ("수익", "비용"),
    ("취득", "처분"),
    ("현금유입", "현금유출"),
    ("손상차손", "손상차손환입"),
    ("충당부채 설정", "충당부채 환입"),
    ("자산 증가", "부채 증가"),  # ANTONYM_PAIRS의 "자산 증가/감소"와 대비되는 회계 항목간 반의어
    ("감가상각", "감가상각 환입"),
    ("미지급비용", "미수수익"),
    ("선급비용", "선수수익"),
    ("대변 기입", "차변 기입"),
    ("수익 인식", "비용 인식"),
    ("자본 증가", "자본 감소"),
    ("영업이익 증가", "영업이익 감소"),
]

def mine_confusable_pairs_from_data(triplets_path, n=20, seed=0):
    """실제 마이닝된 hard negative에서 후보 용어 쌍을 자동 추출 (보조용).

    문서 전체가 아니라 짧은 제목/핵심 어구가 필요하므로, 여기서는
    positive/negative 문서의 첫 줄(보통 조항 제목)을 용어처럼 사용합니다.
    완벽하지 않지만, 사람이 만든 목록의 편향을 보완하는 데이터 기반 대안입니다.
    """
    import json
    import random

    rows = [json.loads(l) for l in open(triplets_path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)

    pairs = []
    seen = set()
    for r in rows:
        pos_title = r["positive"].split("\n")[0].strip()[:30]
        neg_title = r["negatives"][0]["text"].split("\n")[0].strip()[:30]
        key = tuple(sorted([pos_title, neg_title]))
        if pos_title and neg_title and pos_title != neg_title and key not in seen:
            seen.add(key)
            pairs.append((pos_title, neg_title))
        if len(pairs) >= n:
            break
    return pairs

# ---------------------------------------------------------------------------
# 회계 도메인 반의어 자동 추출
# ---------------------------------------------------------------------------
ANTONYM_MORPHEME_PAIRS = [
    ("차변", "대변"),
    ("증가", "감소"),
    ("유입", "유출"),
    ("취득", "처분"),
    ("설정", "환입"),
    ("상각", "환입"),
    ("인식", "제거"),
    ("자산", "부채"),
    ("수익", "비용"),
    ("대여", "차입"),
    ("매입", "매출"),
    ("지급", "수취"),
    ("선급", "선수"),
    ("미지급", "미수"),
]


def mine_accounting_antonym_pairs_from_data(triplets_path, n=15, seed=0):
    import json
    import random

    rows = [json.loads(l) for l in open(triplets_path, encoding="utf-8")]

    titles = set()
    for r in rows:
        titles.add(_clean_title(r["positive"].split("\n")[0]))
        for neg in r.get("negatives", []):
            titles.add(_clean_title(neg["text"].split("\n")[0]))
    titles.discard("")

    titles_list = list(titles)
    random.Random(seed).shuffle(titles_list)

    pairs = []
    seen = set()
    for title in titles_list:
        for morph_a, morph_b in ANTONYM_MORPHEME_PAIRS:
            for src, dst in [(morph_a, morph_b), (morph_b, morph_a)]:
                if src in title:
                    candidate = title.replace(src, dst)
                    if candidate != title and candidate in titles:
                        key = tuple(sorted([title, candidate]))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((title, candidate))
        if len(pairs) >= n:
            break

    return pairs[:n]


# ---------------------------------------------------------------------------
# 회계 도메인 유의어 자동 추출
# ---------------------------------------------------------------------------
import re

SYNONYM_CONTAINMENT_MAX_EXTRA_CHARS = 8
SYNONYM_MIN_BASE_LEN = 3  # 기준 용어(t1) 최소 길이 -- 너무 짧으면 공통 접두사 등 노이즈에 걸림


def _clean_title(title: str) -> str:
    """'[제목] 실제내용' 같은 데이터 포맷 접두사를 제거."""
    return re.sub(r"^\[?제목\]?\s*", "", title).strip()


def mine_accounting_synonym_pairs_from_data(triplets_path, n=15, seed=0):
    """실제 학습 데이터에서 포함관계 기반으로 유의어 후보 쌍을 자동 추출합니다.
    ...(docstring 동일)...
    """
    import json
    import random

    rows = [json.loads(l) for l in open(triplets_path, encoding="utf-8")]
    titles = set()
    for r in rows:
        titles.add(_clean_title(r["positive"].split("\n")[0]))
        for neg in r.get("negatives", []):
            titles.add(_clean_title(neg["text"].split("\n")[0]))
    titles.discard("")  # 정제 후 빈 문자열 방지

    titles_list = list(titles)
    random.Random(seed).shuffle(titles_list)

    pairs = []
    seen = set()
    for t1 in titles_list:
        if len(t1) < SYNONYM_MIN_BASE_LEN:
            continue
        for t2 in titles:
            if t1 == t2 or t1 not in t2:
                continue
            extra = len(t2) - len(t1)
            if extra <= 0 or extra > SYNONYM_CONTAINMENT_MAX_EXTRA_CHARS:
                continue
            suffix = t2.replace(t1, "", 1)
            if any(m in suffix for pair in ANTONYM_MORPHEME_PAIRS for m in pair):
                continue
            key = tuple(sorted([t1, t2]))
            if key not in seen:
                seen.add(key)
                pairs.append((t1, t2))
        if len(pairs) >= n:
            break

    return pairs[:n]


# ---------------------------------------------------------------------------
# 자동 추출 결과 저장/로드 (사람 검수 워크플로우용)
# ---------------------------------------------------------------------------
def save_auto_pairs(pairs, category, output_path):
    """자동 추출된 쌍을 JSON으로 저장. reviewed=False로 시작하며,
    검수 후 사람이 직접 true로 바꾸거나 잘못된 쌍은 지우고 사용합니다."""
    import json
    data = [{"term_a": a, "term_b": b, "category": category, "reviewed": False} for a, b in pairs]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {output_path} ({len(data)}쌍, 전부 미검수 상태)")


def load_auto_pairs(path, only_reviewed=True):
    """save_auto_pairs로 저장된 JSON을 (a, b) 튜플 리스트로 로드.
    only_reviewed=True(기본값)면 검수 완료(reviewed=true)된 쌍만 반환합니다."""
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if only_reviewed:
        data = [d for d in data if d.get("reviewed")]
    return [(d["term_a"], d["term_b"]) for d in data]