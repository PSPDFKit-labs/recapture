"""One version everywhere: the package, the plugin and the skill's install script."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_versions_match():
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())["version"]
    script = re.search(r'^VERSION="([^"]+)"', (ROOT / "skills/recapture/scripts/install.sh").read_text(), re.M).group(1)
    assert package == plugin == script


def test_skill_points_at_files_that_exist():
    skill = (ROOT / "skills/recapture/SKILL.md").read_text()
    assert skill.startswith("---\nname: recapture\ndescription: ")
    for ref in re.findall(r"\]\((references/[^)]+)\)", skill):
        assert (ROOT / "skills/recapture" / ref).is_file(), ref
    assert (ROOT / "skills/recapture/scripts/install.sh").is_file()
