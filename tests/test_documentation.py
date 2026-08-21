from __future__ import annotations

import re
from pathlib import Path

DOC_PATHS = [Path("AGENTS.md"), Path("README.md"), *sorted(Path("docs").rglob("*.md"))]


def test_documentation_avoids_unsupported_latex_constructs() -> None:
    unsupported = (
        r"\operatorname",
        r"\DeclareMathOperator",
        r"\begin{aligned}",
        r"\end{aligned}",
    )
    violations = [
        f"{path}: {macro}"
        for path in DOC_PATHS
        for macro in unsupported
        if macro in path.read_text(encoding="utf-8")
    ]
    assert not violations, "unsupported LaTeX constructs:\n" + "\n".join(violations)


def test_display_math_lines_do_not_look_like_markdown_lists() -> None:
    violations: list[str] = []
    list_marker = re.compile(r"^[ ]{0,3}[+*-][ ]")

    for path in DOC_PATHS:
        in_display_math = False
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line == "$$":
                in_display_math = not in_display_math
            elif in_display_math and list_marker.match(line):
                violations.append(f"{path}:{line_number}: {line}")

    assert not violations, "Markdown list markers inside display math:\n" + "\n".join(
        violations
    )
