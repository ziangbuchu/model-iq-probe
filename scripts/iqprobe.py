#!/usr/bin/env python3
"""25 题知识探针：判断 API / Codex 背后的模型是否被降智（被换成知识更旧的模型）。

题目按"各代 GPT 的知识截止日期"分段。模型能答对的最晚一段 ≈ 它真实的知识边界；
声称是 GPT-6 却答不出 2026 年的题，就说明请求被路由到了更旧的模型。
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = (ROOT / "assets" / "quiz_prompt.txt").read_text(encoding="utf-8")
KEY = json.loads((ROOT / "assets" / "answer_key.json").read_text(encoding="utf-8"))
QUESTIONS = KEY["questions"]
SECTIONS = KEY["sections"]


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "", s)


# ---------- 判分 ----------

def parse_answers(text):
    answers = {}
    for line in text.splitlines():
        line = line.replace("*", "").replace("`", "")
        m = re.match(r"^\W*(Q\d{2})\s*[｜|:：]\s*(.*?)\s*$", line)
        if m and m.group(1) in QUESTIONS:
            answers[m.group(1)] = m.group(2)  # 取最后一次出现：粘贴的整段对话里题目在前、回答在后
    return answers


def grade_one(qid, ans):
    if ans is None:
        return "missing"
    n = norm(ans)
    if not n or norm("不清楚") in n:
        return "unknown"
    groups = QUESTIONS[qid]["match"]
    return "correct" if all(any(norm(a) in n for a in g) for g in groups) else "wrong"


def expected_cutoff_for(model):
    for pattern, cutoff in KEY["expected_cutoff_by_model"]:
        if model and re.search(pattern, model, re.I):
            return cutoff
    return None


def section_passed(correct, total):
    return correct >= max(1, math.ceil(total / 2))


def evaluate(text, expected_cutoff):
    answers = parse_answers(text)
    status = {q: grade_one(q, answers.get(q)) for q in QUESTIONS}
    sections = []
    for sec in SECTIONS:
        qs = sec["questions"]
        c = sum(status[q] == "correct" for q in qs)
        sections.append({**sec, "correct": c, "total": len(qs),
                         "unknown": sum(status[q] == "unknown" for q in qs),
                         "wrong": sum(status[q] == "wrong" for q in qs),
                         "missing": sum(status[q] == "missing" for q in qs)})
    passed = [s for s in sections if section_passed(s["correct"], s["total"])]
    horizon = passed[-1]["label"] if passed else None
    correct_dates = [QUESTIONS[q]["date"] for q in QUESTIONS if status[q] == "correct"]

    verdict, target = None, None
    if expected_cutoff:
        eligible = [s for s in sections
                    if max(QUESTIONS[q]["date"] for q in s["questions"]) <= expected_cutoff]
        target = eligible[-1] if eligible else None
        newer_pass = any(section_passed(s["correct"], s["total"])
                         for s in sections if s not in eligible)
        if target is None:
            verdict = "无法判断"
        elif section_passed(target["correct"], target["total"]):
            verdict = "不符（知识比声称的新）" if newer_pass else "正常"
        elif target["correct"] <= 1 or target["total"] < 5:
            verdict = "降智"
        else:
            verdict = "存疑"
    return {"answers": answers, "status": status, "sections": sections,
            "horizon": horizon, "latest_correct_date": max(correct_dates) if correct_dates else None,
            "target_section": target["label"] if target else None, "verdict": verdict,
            "parsed": len(answers)}


# ---------- 调用 ----------

def load_codex_provider():
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    cfg = tomllib.loads((home / "config.toml").read_text())
    name = cfg.get("model_provider", "openai")
    prov = cfg.get("model_providers", {}).get(name, {})
    key = None
    if prov.get("env_key"):
        key = os.environ.get(prov["env_key"])
    if not key and (home / "auth.json").exists():
        key = json.loads((home / "auth.json").read_text()).get("OPENAI_API_KEY")
    return {"base_url": prov.get("base_url", "https://api.openai.com/v1"),
            "headers": prov.get("http_headers", {}),
            "api": "chat" if prov.get("wire_api") == "chat" else "responses",
            "model": cfg.get("model"), "effort": cfg.get("model_reasoning_effort"), "key": key}


# 很多中转挂在 Cloudflare 后面，会拦 Python-urllib 这类默认 UA（HTTP 403 / error code 1010），
# 有的只放行 Codex 客户端，所以默认按 codex exec 的样子发请求头。
DEFAULT_UA = "codex_exec/0.162.1 (Ubuntu 22.4.0; x86_64)"


def call_api(a, prompt):
    base = a.base_url.rstrip("/")
    # 一律流式：部分中转只收 stream=true，流式也更接近 Codex 的真实请求
    if a.api == "responses":
        url, body = base + "/responses", {"model": a.model, "input": prompt, "stream": True}
        if a.effort:
            body["reasoning"] = {"effort": a.effort}
    else:
        url = base + "/chat/completions"
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "stream": True}
        if a.effort:
            body["reasoning_effort"] = a.effort
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
               "User-Agent": a.user_agent, "originator": "codex_exec", **a.headers}
    if a.key:
        headers["Authorization"] = f"Bearer {a.key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            if "event-stream" not in r.headers.get("Content-Type", ""):
                return parse_json_reply(a.api, json.loads(r.read().decode()))
            return read_sse(a.api, r)
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
        if e.code == 403:
            msg += "\n（可能是 Cloudflare 按 User-Agent 拦截，可用 --user-agent 换一个再试）"
        raise RuntimeError(msg) from None


def parse_json_reply(api, data):
    if api == "chat":
        return data["choices"][0]["message"]["content"], data.get("model")
    if data.get("output_text"):
        return data["output_text"], data.get("model")
    parts = [c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    return "\n".join(parts), data.get("model")


def read_sse(api, r):
    out, model = [], None
    for raw in r:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            e = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if api == "chat":
            model = e.get("model") or model
            for ch in e.get("choices", []):
                out.append((ch.get("delta") or {}).get("content") or "")
            continue
        t = e.get("type")
        if t == "response.output_text.delta":
            out.append(e.get("delta", ""))
        elif t in ("response.created", "response.completed"):
            model = e.get("response", {}).get("model") or model
        elif t in ("error", "response.failed"):
            raise RuntimeError(f"流式返回错误：{payload[:300]}")
    return "".join(out), model


def call_codex(a, prompt):
    with tempfile.TemporaryDirectory(prefix="iqprobe-") as d:
        out = Path(d) / "last.txt"
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-C", d, "-s", "read-only",
               "-c", 'web_search="disabled"', "-o", str(out)]
        if a.model:
            cmd += ["-m", a.model]
        if a.effort:
            cmd += ["-c", f'model_reasoning_effort="{a.effort}"']
        cmd.append("-")
        p = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=a.timeout)
        if p.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex exec 失败 (rc={p.returncode}): {p.stderr.strip()[-300:]}")
        return out.read_text(), None  # codex exec 不回报服务端模型名


def one_run(a, i):
    t0 = time.time()
    try:
        text, served = (call_codex if a.via == "codex" else call_api)(a, PROMPT)
        return {"run": i, "seconds": round(time.time() - t0, 1), "served_model": served, "text": text}
    except Exception as e:
        return {"run": i, "seconds": round(time.time() - t0, 1), "error": str(e)}


# ---------- 输出 ----------

def report(r, show_raw):
    head = f"第 {r['run']} 轮" if r.get("run") else "判分"
    if r.get("error"):
        print(f"\n## {head}：调用失败（{r['seconds']}s）\n{r['error']}")
        return
    ev = r["eval"]
    extra = f"，{r['seconds']}s" if "seconds" in r else ""
    if r.get("served_model"):
        extra += f"，返回 model={r['served_model']}"
    print(f"\n## {head}{extra}")
    if ev["parsed"] < 20:
        print(f"⚠ 只解析到 {ev['parsed']}/25 行答案，模型可能没按格式作答，结论不可靠")
    for s in ev["sections"]:
        mark = "✓" if section_passed(s["correct"], s["total"]) else "✗"
        bits = [f"对 {s['correct']}/{s['total']}"]
        for k, zh in (("unknown", "不清楚"), ("wrong", "答错"), ("missing", "缺失")):
            if s[k]:
                bits.append(f"{zh} {s[k]}")
        print(f"  {mark} {s['label']:<18} (≤{s['cutoff']})  " + "，".join(bits))
    wrong = [q for q, st in ev["status"].items() if st == "wrong"]
    for q in wrong:
        print(f"    {q} 答「{ev['answers'][q]}」，参考：{QUESTIONS[q]['answer']}")
    print(f"  知识边界：表现接近 {ev['horizon'] or '（无一段过半）'}；最晚答对 {ev['latest_correct_date']}")
    if ev["verdict"]:
        print(f"  判定：{ev['verdict']}（看「{ev['target_section']}」段）")
    if show_raw:
        print("  --- 原始回答 ---\n" + r["text"])


def main():
    p = argparse.ArgumentParser(description="25 题知识探针：检测模型 API 是否降智")
    p.add_argument("--from-codex", action="store_true", help="读取 ~/.codex 的中转地址、密钥、默认模型和推理档位")
    p.add_argument("--via", choices=["api", "codex"], default="api", help="api=直接调接口；codex=走 codex exec 客户端")
    p.add_argument("--base-url")
    p.add_argument("--key-env", default="OPENAI_API_KEY", help="存放 API 密钥的环境变量名")
    p.add_argument("--api", choices=["responses", "chat"])
    p.add_argument("-m", "--model")
    p.add_argument("--effort", help="推理档位，如 low/medium/high/xhigh")
    p.add_argument("--expect-cutoff", help="期望的知识截止日期 YYYY-MM-DD；不给则按模型名推断")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--user-agent", default=DEFAULT_UA, help="请求的 User-Agent，默认模仿 codex exec")
    p.add_argument("--grade", metavar="FILE", help="只给已有回答判分（- 表示 stdin），不调用模型")
    p.add_argument("--save", metavar="DIR", help="把每轮原始回答和判分存成 JSON")
    p.add_argument("--show-raw", action="store_true")
    a = p.parse_args()

    if a.grade:
        text = sys.stdin.read() if a.grade == "-" else Path(a.grade).read_text()
        exp = a.expect_cutoff or expected_cutoff_for(a.model)
        report({"eval": evaluate(text, exp)}, a.show_raw)
        return

    a.headers = {}
    if a.from_codex or a.via == "codex":
        c = load_codex_provider()
        a.base_url = a.base_url or c["base_url"]
        a.api = a.api or c["api"]
        a.model = a.model or c["model"]
        a.effort = a.effort or c["effort"]
        a.headers = c["headers"]
        a.key = c["key"]
    else:
        a.key = os.environ.get(a.key_env)
        if not a.base_url or not a.model:
            p.error("直接测 API 需要 --base-url 和 -m（或用 --from-codex）")
        if not a.key:
            p.error(f"环境变量 {a.key_env} 里没有密钥")
    a.api = a.api or "responses"
    exp = a.expect_cutoff or expected_cutoff_for(a.model)

    target = "codex exec" if a.via == "codex" else f"{a.base_url} [{a.api}]"
    print(f"# 降智探针：model={a.model} effort={a.effort or '默认'} via {target}")
    print(f"  期望知识截止：{exp or '未知（只报知识边界，不下结论）'}；共 {a.runs} 轮")

    with ThreadPoolExecutor(max_workers=max(1, min(a.parallel, a.runs))) as ex:
        results = list(ex.map(lambda i: one_run(a, i), range(1, a.runs + 1)))
    for r in results:
        if "text" in r:
            r["eval"] = evaluate(r["text"], exp)
        report(r, a.show_raw)

    ok = [r for r in results if "eval" in r]
    if a.runs > 1 and ok and exp:
        tally = {}
        for r in ok:
            tally[r["eval"]["verdict"]] = tally.get(r["eval"]["verdict"], 0) + 1
        print("\n# 汇总：" + "，".join(f"{k} {v}/{len(ok)}" for k, v in tally.items())
              + (f"，失败 {len(results) - len(ok)}" if len(ok) < len(results) else ""))

    if a.save:
        d = Path(a.save)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"iqprobe-{datetime.now():%Y%m%d-%H%M%S}.json"
        meta = {"model": a.model, "effort": a.effort, "via": a.via, "base_url": a.base_url,
                "api": a.api, "expected_cutoff": exp}
        f.write_text(json.dumps({"meta": meta, "runs": results}, ensure_ascii=False, indent=2))
        print(f"\n已保存：{f}")


if __name__ == "__main__":
    main()
