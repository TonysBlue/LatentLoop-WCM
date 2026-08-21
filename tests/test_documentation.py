from __future__ import annotations

from pathlib import Path


def test_documentation_avoids_unsupported_latex_constructs() -> None:
    paths = [Path("AGENTS.md"), Path("README.md"), *sorted(Path("docs").rglob("*.md"))]
    unsupported = (
        r"\operatorname",
        r"\DeclareMathOperator",
        r"\begin{aligned}",
        r"\end{aligned}",
    )
    violations = [
        f"{path}: {macro}"
        for path in paths
        for macro in unsupported
        if macro in path.read_text(encoding="utf-8")
    ]
    assert not violations, "unsupported LaTeX constructs:\n" + "\n".join(violations)
