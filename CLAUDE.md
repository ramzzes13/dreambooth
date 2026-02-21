# Overnight Execution Rules

## CRITICAL CONSTRAINTS
- ALL code must be REAL and WORKING. Never mock, stub, or fake any functionality.
- Do NOT modify any files outside `/home/mekashirskiy/rom4ik/dreambooth/`
- Do NOT kill any processes you did not create. There are sglang servers and other services running.
- Commit AND push frequently with meaningful messages after each milestone. Always `git add . && git commit -m "..." && git push` after committing.
- This is a full autonomous overnight execution. Keep working until ALL goals from the research plan are reached.
- If a dependency install fails, try alternative approaches. Do not stop.
- If training takes time, wait for it. Do not skip or mock results.
- All models, data downloads, and training runs must be real.
- Use GPU resources available on the machine. Check with `nvidia-smi` before starting.
- Do NOT install packages into the project directory. Use pip install --user or system pip.

## Progress Tracking
- Read `progress.txt` at the start to see what previous iterations accomplished.
- Before exiting, ALWAYS update `progress.txt` with what you completed and what remains.
- If ALL goals are fully achieved, include `<promise>COMPLETE</promise>` in your output AND in progress.txt.

## Project Goal
Implement ModularBooth from `dreambooth_research_plan.md`:
1. Blockwise-parameterized LoRA for DiT architectures
2. Contrastive disentanglement loss
3. Cross-attention masking inference for multi-subject composition
4. Extended multi-subject benchmark evaluation

## PAPER SYNTHESIS (MANDATORY)
You MUST write a full research paper (LaTeX, ICML format). This is not optional.

### Paper Requirements:
- **Format**: ICML 2025/2026 LaTeX template (use `icml2025.sty` or download from ICML website)
- **Length**: 8-10 pages main body + unlimited appendix
- **Quality**: ICML A* level. No neuroslop (no "delve", "leverage", "in this paper we", "it is worth noting"). Write like a real researcher.
- **Citations**: Use BibTeX. Every citation must be REAL and VERIFIED - check that the paper actually exists, verify authors, year, venue. Do NOT hallucinate citations. Cross-reference with the PDF `dreambooth.pdf` in this folder for the original paper's references.
- **Figures**: All figures must use real experimental results. Generate matplotlib/tikz plots from actual data. No placeholder figures.
- **Results**: All numbers in tables/text must come from actual experiments run in this project. Cross-verify with output logs.
- **GitHub Link**: Include `https://github.com/ramzzes13/dreambooth` in the paper (abstract footnote or after title).
- **Sections**: Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion, References
- **Verification**: After writing, re-read the paper and verify every claim against actual code/results. Fix any inconsistencies.

### Writing Style:
- Concise, precise, scientific English
- No filler sentences or vague claims
- Every paragraph must convey information
- Compare fairly with baselines
- Acknowledge limitations honestly
