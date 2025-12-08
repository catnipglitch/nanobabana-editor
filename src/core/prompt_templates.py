"""
Prompt Templates

プロンプトテンプレートの読み込みと管理を行うモジュール。
app_config.toml からテンプレートを読み込み、UI で使用できる形式で提供する。

テンプレートはプロンプト文字列とカテゴリのみを管理し、モデル選択は別の責務とする。
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Python 3.11+ では標準ライブラリの tomllib を使用
if sys.version_info >= (3, 11):
    import tomllib
else:
    # Python 3.10以下の場合は tomli を使用
    try:
        import tomli as tomllib
    except ImportError:  # pragma: no cover - 実行環境では3.11+を想定
        raise ImportError(
            "Python 3.10 以下では tomli パッケージが必要です。\n"
            "インストール: uv add tomli"
        )

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """プロンプトテンプレート"""

    name: str
    prompt: str
    category: str = "text_to_image"

    def to_dict(self) -> Dict[str, str]:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "category": self.category,
        }


class PromptTemplateManager:
    """プロンプトテンプレートの管理クラス"""

    def __init__(self, template_file: Path | None = None):
        """
        Args:
            template_file: テンプレートファイルのパス（オプション）
                          - None の場合: マルチファイルモード（config/templates/*.toml から読み込み）
                          - 指定された場合: 単一ファイルモード（指定されたファイルのみ読み込み）
        """
        self.template_file = template_file  # None の場合はマルチファイルモード
        self.templates: List[PromptTemplate] = []
        self._load_templates()

    def _load_templates(self) -> None:
        """
        テンプレートファイルを読み込む（マルチファイル対応）

        読み込み優先順位:
        1. config/templates/*.toml (推奨、カテゴリ別分割ファイル)
        2. app_config.toml (後方互換性のためサポート継続)
        3. デフォルトテンプレート (フォールバック)
        """
        # 単一ファイルモード（template_file が指定されている場合）
        if self.template_file is not None:
            logger.info("Single-file mode: %s", self.template_file)
            if self.template_file.exists():
                self._load_template_file(self.template_file)
            else:
                logger.warning("Template file not found: %s fallback=defaults", self.template_file)
                self.templates = self._get_default_templates()
            return

        # マルチファイルモード
        project_root = Path(__file__).parent.parent.parent
        template_dir = project_root / "config" / "templates"
        legacy_file = project_root / "app_config.toml"

        # ステップ1: config/templates/ ディレクトリから読み込み
        if template_dir.exists() and template_dir.is_dir():
            logger.info("Loading templates from directory: %s", template_dir)
            self._load_template_directory(template_dir)

        # ステップ2: app_config.toml から読み込み（後方互換性）
        if not self.templates and legacy_file.exists():
            logger.info("Loading templates from legacy file: %s", legacy_file)
            self._load_template_file(legacy_file)

        # ステップ3: デフォルトテンプレート（フォールバック）
        if not self.templates:
            logger.warning("No templates found, using defaults")
            self.templates = self._get_default_templates()

        logger.info("Loaded templates: count=%d", len(self.templates))
        self._check_duplicates()

    def _load_template_directory(self, directory: Path) -> None:
        """
        ディレクトリ内の全 .toml ファイルからテンプレートを読み込む

        Args:
            directory: テンプレートディレクトリのパス
        """
        toml_files = sorted(directory.glob("*.toml"))

        if not toml_files:
            logger.warning("No .toml files found in: %s", directory)
            return

        logger.debug("Found template files: count=%d", len(toml_files))

        for toml_file in toml_files:
            logger.debug("Loading template file: %s", toml_file.name)
            self._load_template_file(toml_file)

    def _load_template_file(self, file_path: Path) -> None:
        """
        単一の .toml ファイルからテンプレートを読み込む

        Args:
            file_path: テンプレートファイルのパス
        """
        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            if "templates" not in data:
                logger.warning("No [[templates]] section in: %s", file_path.name)
                return

            loaded_count = 0
            for template_data in data["templates"]:
                template = PromptTemplate(
                    name=template_data.get("name", "Unknown"),
                    prompt=template_data.get("prompt", ""),
                    category=template_data.get("category", "text_to_image"),
                )
                self.templates.append(template)
                loaded_count += 1
                logger.debug(
                    "Template loaded: file=%s name=%s category=%s",
                    file_path.name, template.name, template.category
                )

            logger.info("Loaded templates from file: file=%s count=%d",
                       file_path.name, loaded_count)

        except Exception as exc:
            logger.exception("Error loading template file %s: %s", file_path.name, exc)

    def _check_duplicates(self) -> None:
        """テンプレート名の重複をチェックし、警告を出力"""
        seen_names = {}
        for template in self.templates:
            if template.name in seen_names:
                logger.warning(
                    "Duplicate template name: name=%s first_category=%s current_category=%s",
                    template.name, seen_names[template.name], template.category
                )
            else:
                seen_names[template.name] = template.category

    def _get_default_templates(self) -> List[PromptTemplate]:
        """デフォルトのテンプレートを返す"""
        return [
            PromptTemplate(
                name="風景 - 夕焼けの山",
                prompt="A beautiful sunset over mountains",
                category="text_to_image",
            ),
            PromptTemplate(
                name="動物 - 猫",
                prompt="A cute cat sleeping on a cozy sofa",
                category="text_to_image",
            ),
        ]

    def get_template_names(self) -> List[str]:
        """テンプレート名のリストを取得"""
        return [template.name for template in self.templates]

    def get_template_by_name(self, name: str) -> Optional[PromptTemplate]:
        """
        名前からテンプレートを取得

        Args:
            name: テンプレート名

        Returns:
            見つかった場合は PromptTemplate、見つからない場合は None
        """
        for template in self.templates:
            if template.name == name:
                logger.debug("Template found: name=%s category=%s", template.name, template.category)
                return template
        logger.warning("Template not found: name=%s", name)
        return None

    def get_all_templates(self) -> List[PromptTemplate]:
        """全テンプレートを取得"""
        return self.templates

    def get_template_choices(self) -> List[str]:
        """
        Gradio の Dropdown で使用するための選択肢リストを取得
        先頭に「選択してください」を追加
        """
        return ["選択してください"] + self.get_template_names()

    def get_templates_by_category(self, category: str) -> List[PromptTemplate]:
        """
        カテゴリでフィルタリングしたテンプレートのリストを取得

        Args:
            category: テンプレートカテゴリ (text_to_image, image_edit, multi_image)

        Returns:
            指定されたカテゴリのテンプレートリスト
        """
        return [t for t in self.templates if t.category == category]

    def get_template_choices_for_tab(self, tab_key: str) -> List[str]:
        """
        タブキーに応じたテンプレート選択肢リストを取得

        Args:
            tab_key: タブキー (gemini_gen01, gemini_edit01, gemini_edit02)

        Returns:
            Gradio の Dropdown で使用する選択肢リスト
        """
        category_mapping = {
            "gemini_gen01": "text_to_image",
            "gemini_edit01": "image_edit",
            "gemini_edit02": "multi_image",
            "gemini_edit03": "multiturn_edit",
        }
        category = category_mapping.get(tab_key, "text_to_image")
        templates = self.get_templates_by_category(category)
        logger.debug(
            "Template choices for tab: tab_key=%s category=%s count=%d",
            tab_key, category, len(templates)
        )
        return ["選択してください"] + [t.name for t in templates]
