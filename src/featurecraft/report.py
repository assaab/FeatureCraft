"""Report generation utilities for FeatureCraft."""

from __future__ import annotations

import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import FeatureCraftConfig
from .logging import get_logger
from .types import DatasetInsights

logger = get_logger(__name__)


class ReportBuilder:
    """Build HTML reports from insights."""

    def __init__(self, templates_dir: str = "templates") -> None:
        """Initialize with templates directory."""
        self.env = Environment(
            loader=FileSystemLoader(templates_dir), autoescape=select_autoescape(["html", "xml"])
        )

    def build(self, insights: DatasetInsights, out_path: str, cfg: FeatureCraftConfig) -> str:
        """Build and save HTML report."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        template = self.env.get_template("report.html.j2")
        html = template.render(
            insights=insights,
            profiles=insights.profiles,
            issues=insights.issues,
            figures=insights.figures,
            summary=insights.summary,
            cfg=cfg,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Saved report to %s", out_path)
        return out_path
