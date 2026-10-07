# model-iq-probe

用 25 道按知识截止日期分段的题目，检测模型 API、中转站或 Codex 里的模型是否“降智”（声称的模型被偷换成知识更旧的模型）。Claude Code / Codex 的 Skill，也可以单独当命令行脚本用。用法和判定规则见 [SKILL.md](SKILL.md)。

题目来自 https://linux.do/t/topic/2978609 的回复。

## 安装

```bash
git clone https://github.com/ziangbuchu/model-iq-probe ~/.claude/skills/model-iq-probe
```

只依赖 Python 3.11+ 标准库。单独使用：

```bash
python3 scripts/iqprobe.py --from-codex --runs 3
python3 scripts/iqprobe.py --base-url URL -m MODEL --key-env OPENAI_API_KEY --runs 3
```
