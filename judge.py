"""
3단계: Gemini 2.5 Pro judge로 faithfulness 등 생성측 지표 산출.

Faithfulness (RAGAS 정의, 2단계 구현):
  (1) 답변을 원자적 주장(atomic statement)으로 분해
  (2) 각 주장이 context에서 직접 추론 가능한지 0/1 판정
  faithfulness = 지지된 주장 수 / 전체 주장 수

부가 지표:
  answer_relevancy   : 답변이 질문에 실제로 답하는가 (0~1)
  gold_agreement     : 정답 문서로 만든 답변과 의미가 일치하는가 (0/0.5/1)
                       -> retrieval 실패가 답변을 얼마나 망쳤는지를 직접 보여줌
  abstention_rate    : 기권 비율

주의: 기권 답변은 faithfulness가 자동으로 1.0이 됩니다(문서에 반하는 주장이 없으므로).
평균에 섞으면 나쁜 retriever가 1등이 됩니다. 그래서 기본적으로 분리 집계합니다.
"""
import argparse
import json

from tqdm import tqdm

from config import DEFAULT_MODELS, JUDGE_MODEL, RUN_DIR
from gemini_client import call

# ---------------------------------------------------------------- prompts
DECOMPOSE_SYS = """당신은 텍스트 분석기입니다. 주어진 답변을 검증 가능한 최소 단위의
독립적인 주장(statement)들로 분해하세요. 각 주장은 대명사 없이 그 자체로 이해 가능해야 합니다.
JSON만 출력하세요: {"statements": ["...", "..."]}"""

DECOMPOSE = """[질문]
{question}

[답변]
{answer}"""

VERDICT_SYS = """당신은 엄격한 사실 검증기입니다. 주어진 컨텍스트만을 근거로 각 주장을 판정하세요.
- verdict 1: 컨텍스트에서 직접 도출되거나 명시된 주장
- verdict 0: 컨텍스트에 없거나, 컨텍스트와 모순되거나, 외부 지식이 필요한 주장
상식이나 사전 지식으로 보충하지 마세요. 컨텍스트에 없으면 0입니다.
JSON만 출력하세요: {"verdicts": [{"statement": "...", "verdict": 0 or 1, "reason": "한 문장"}]}"""

VERDICT = """[컨텍스트]
{contexts}

[검증할 주장들]
{statements}"""

RELEVANCY_SYS = """답변이 질문에 얼마나 직접적으로 답하는지 평가하세요. 사실 여부는 보지 마세요.
1.0=질문에 온전히 답함, 0.5=부분적으로만 답하거나 우회함, 0.0=무관하거나 답을 회피함.
JSON만 출력하세요: {"score": 1.0, "reason": "한 문장"}"""

AGREE_SYS = """두 답변이 같은 질문에 대해 의미적으로 동일한 정보를 전달하는지 판정하세요.
1.0=핵심 내용이 일치, 0.5=일부만 일치하거나 한쪽이 불완전, 0.0=불일치하거나 한쪽이 답을 못 함.
표현 차이는 무시하고 내용만 보세요.
JSON만 출력하세요: {"score": 1.0, "reason": "한 문장"}"""


# ---------------------------------------------------------------- metrics
def faithfulness(question, answer, contexts):
    st = call(DECOMPOSE.format(question=question, answer=answer),
              model=JUDGE_MODEL, system=DECOMPOSE_SYS, as_json=True)
    statements = [s for s in st.get("statements", []) if s.strip()]
    if not statements:
        return None, []                    # 평균에서 제외 (0점 아님)

    ctx = "\n\n".join(f"<문서 {i+1}>\n{c}" for i, c in enumerate(contexts))
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
    vd = call(VERDICT.format(contexts=ctx, statements=numbered),
              model=JUDGE_MODEL, system=VERDICT_SYS, as_json=True)
    verdicts = vd.get("verdicts", [])
    if not verdicts:
        return None, []
    supported = sum(1 for v in verdicts if int(v.get("verdict", 0)) == 1)
    return supported / len(verdicts), verdicts


def _score(prompt, system):
    r = call(prompt, model=JUDGE_MODEL, system=system, as_json=True)
    return float(r.get("score", 0.0)), r.get("reason", "")


def judge_model(name: str, with_gold: bool):
    run_dir = RUN_DIR / name
    answers = json.load(open(run_dir / "answers.json", encoding="utf-8"))
    gold = {}
    if with_gold and (run_dir / "answers_gold.json").exists():
        gold = json.load(open(run_dir / "answers_gold.json", encoding="utf-8"))

    results = {}
    for qid, a in tqdm(answers.items(), desc=f"judge[{name}]"):
        row = {"abstained": a["abstained"],
               "n_gold_in_context": a["n_gold_in_context"]}

        f, verdicts = faithfulness(a["query"], a["answer"], a["contexts"])
        row["faithfulness"] = f
        row["verdicts"] = verdicts

        row["answer_relevancy"], row["relevancy_reason"] = _score(
            f"[질문]\n{a['query']}\n\n[답변]\n{a['answer']}", RELEVANCY_SYS)

        if qid in gold:
            row["gold_agreement"], row["agreement_reason"] = _score(
                f"[질문]\n{a['query']}\n\n[답변 A (정답 문서 기반)]\n"
                f"{gold[qid]['answer']}\n\n[답변 B (검색 문서 기반)]\n{a['answer']}",
                AGREE_SYS)
        results[qid] = row

    json.dump(results, open(run_dir / "judge.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 요약: 기권 건은 분리
    ans = [r for r in results.values() if not r["abstained"]]
    def avg(rows, k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return sum(v) / len(v) if v else float("nan")

    summ = {
        "n": len(results),
        "abstention_rate": sum(r["abstained"] for r in results.values()) / len(results),
        "faithfulness_answered": avg(ans, "faithfulness"),
        "faithfulness_all": avg(list(results.values()), "faithfulness"),
        "answer_relevancy": avg(list(results.values()), "answer_relevancy"),
        "gold_agreement": avg(list(results.values()), "gold_agreement"),
    }
    json.dump(summ, open(run_dir / "judge_summary.json", "w", encoding="utf-8"),
              indent=2)
    print(f"[{name}] " + "  ".join(f"{k}={v:.3f}" for k, v in summ.items()
                                   if isinstance(v, float)))
    return summ


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--no-gold", action="store_true")
    a = p.parse_args()
    for m in a.models:
        judge_model(m, with_gold=not a.no_gold)