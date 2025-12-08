"""
Gemini Tab

Gemini image generation tab.
"""

import gradio as gr
import logging
import io
import time
import json
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from PIL import Image
from .base_tab import BaseTab
from ...core.generators import GenerationConfig, ModelType
from ...core.tab_specs import TAB_GEMINI
from ...core.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


@dataclass
class GeminiControls:
    model: Any
    template: Any
    prompt: Any
    optimized_prompt: Any
    optimization_level: Any
    optimize_prompt_button: Any
    aspect_ratio: Any
    image_size: Any
    google_search: Any
    gen_button_optimize_and_gen: Any
    gen_button_use_opt: Any
    gen_button_no_opt: Any
    reset_button: Any
    output_image: Any
    output_info: Any
    output_json: Any


TEST_MODEL_NAME = "test-model"
GEMINI_OUTPUT_PREFIX = "gemini_gen"
DEFAULT_IMAGE_SIZE = "1K"

ERROR_API_KEY_NOT_CONFIGURED = """❌ エラー: APIキーが設定されていません

**Settings タブ** でGoogle API Keyを設定してください。

1. Settings タブを開く
2. APIキーを入力
3. 「接続テスト」ボタンで確認
4. 「APIキーを適用」ボタンで適用
"""

WARNING_EMPTY_OPTIMIZED_PROMPT = """⚠️ 警告: 最適化済みプロンプトが空です

元のプロンプトを使用して生成します。
先に「画像生成（プロンプトの最適化を実施してから生成）」を実行してください。
"""

ERROR_OPTIMIZED_PROMPT_HAS_ERROR = """⚠️ エラー: 最適化済みプロンプトにエラーが含まれています

「画像生成（プロンプトの最適化を実施してから生成）」を再実行してください。
"""


class GeminiTab(BaseTab):
    """Gemini image generation tab"""

    def create_ui(self) -> None:
        """Create Gemini tab UI"""
        with gr.Tab(
            TAB_GEMINI.display_name,
            id=TAB_GEMINI.key,
            elem_id=TAB_GEMINI.elem_id,
        ):
            controls = self._build_layout()
            self._bind_events(controls)

    def _build_layout(self) -> GeminiControls:
        with gr.Row():
            with gr.Column(scale=1):
                (
                    gemini_model,
                    gemini_template,
                    gemini_prompt,
                    gemini_optimized_prompt,
                    gemini_optimization_level,
                    gemini_optimize_prompt_button,
                    gemini_aspect_ratio,
                    gemini_image_size,
                    gemini_google_search,
                    gemini_gen_button_optimize_and_gen,
                    gemini_gen_button_use_opt,
                    gemini_gen_button_no_opt,
                    gemini_reset_button,
                ) = self._build_input_column()

            with gr.Column(scale=1):
                gemini_output_image, gemini_output_info, gemini_output_json = (
                    self._build_output_column()
                )

        return GeminiControls(
            model=gemini_model,
            template=gemini_template,
            prompt=gemini_prompt,
            optimized_prompt=gemini_optimized_prompt,
            optimization_level=gemini_optimization_level,
            optimize_prompt_button=gemini_optimize_prompt_button,
            aspect_ratio=gemini_aspect_ratio,
            image_size=gemini_image_size,
            google_search=gemini_google_search,
            gen_button_optimize_and_gen=gemini_gen_button_optimize_and_gen,
            gen_button_use_opt=gemini_gen_button_use_opt,
            gen_button_no_opt=gemini_gen_button_no_opt,
            reset_button=gemini_reset_button,
            output_image=gemini_output_image,
            output_info=gemini_output_info,
            output_json=gemini_output_json,
        )

    def _build_input_column(self):
        # プロンプト入力
        gemini_model = gr.Dropdown(
            label="モデル",
            choices=self.app.gemini_models,
            value=self.app.gemini_models[0] if self.app.gemini_models else None,
            info="Gemini画像生成モデル",
        )

        gemini_template = gr.Dropdown(
            label="プロンプトテンプレート",
            choices=self.app.template_manager.get_template_choices_for_tab(
                "gemini_gen01"
            ),
            value="選択してください",
            info="サンプルプロンプトを選択",
        )

        gemini_prompt = gr.Textbox(
            label="ユーザープロンプト",
            placeholder="生成したい画像を説明してください",
            lines=4,
        )

        # 最適化されたプロンプト表示
        gemini_optimized_prompt = gr.Textbox(
            label="最適化されたプロンプト",
            placeholder="「プロンプト生成」または「編集開始」をクリックすると、最適化されたプロンプトが表示されます",
            lines=5,
            interactive=True,
            info="生成前にプロンプトを確認・編集できます",
        )

        # プロンプト最適化レベル
        gemini_optimization_level = gr.Radio(
            label="プロンプト最適化レベル",
            choices=[
                ("レベル1: 修正のみ（誤字脱字・語順）", 1),
                ("レベル2: 標準最適化（推奨）", 2),
                ("レベル3: 創造的拡張", 3),
            ],
            value=1,
            info="レベル1推奨: Gemini 2.0 Flashで誤字脱字を修正",
        )

        # 最適化プロンプト生成ボタン
        gemini_optimize_prompt_button = gr.Button(
            "最適化プロンプト生成",
            variant="secondary",
            size="sm",
        )

        # 詳細設定
        with gr.Accordion("詳細設定", open=True):
            gemini_aspect_ratio = gr.Radio(
                label="アスペクト比",
                choices=[
                    "1:1",  # 正方形
                    "2:3",  # 縦長（ポートレート）
                    "3:2",  # 横長
                    "3:4",  # 縦長（ポートレート）
                    "4:3",  # 横長
                    "4:5",  # 縦長（ポートレート）
                    "5:4",  # 横長
                    "9:16",  # 縦長（スマホ縦）
                    "16:9",  # 横長（ワイドスクリーン）
                    "21:9",  # 超横長（シネマスコープ）
                ],
                value=self.app.default_aspect_ratio,
                info="生成画像の縦横比（全10種類サポート）",
            )

            gemini_image_size = gr.Radio(
                label="解像度",
                choices=["1K", "2K", "4K"],
                value=DEFAULT_IMAGE_SIZE,
                info="Gemini 3 Pro Image Preview で有効（他モデルでは無視されます）",
            )

            gr.Markdown("**注意**: Geminiは1枚のみ生成します")

        # ツールオプション（Phase 3.0）
        with gr.Accordion("ツールオプション", open=False):
            gemini_google_search = gr.Checkbox(
                label="Google Search",
                value=False,
                info="Google検索でリアルタイム情報を取得（Gemini 3 Pro Image推奨）",
            )

        # ボタン（2行2列レイアウト）
        with gr.Row():
            gemini_gen_button_optimize_and_gen = gr.Button(
                "画像生成（プロンプトの最適化を実施してから生成）",
                variant="primary",
                size="sm",
            )
            gemini_gen_button_use_opt = gr.Button(
                "画像生成（最適化済みプロンプトで生成）",
                variant="secondary",
                size="sm",
            )

        with gr.Row():
            gemini_gen_button_no_opt = gr.Button(
                "画像生成（ユーザープロンプトで生成）",
                variant="secondary",
                size="sm",
            )
            gemini_reset_button = gr.Button(
                "リセット",
                size="sm",
            )

        return (
            gemini_model,
            gemini_template,
            gemini_prompt,
            gemini_optimized_prompt,
            gemini_optimization_level,
            gemini_optimize_prompt_button,
            gemini_aspect_ratio,
            gemini_image_size,
            gemini_google_search,
            gemini_gen_button_optimize_and_gen,
            gemini_gen_button_use_opt,
            gemini_gen_button_no_opt,
            gemini_reset_button,
        )

    def _build_output_column(self):
        # 出力エリア
        gemini_output_image = gr.Image(label="生成された画像", type="pil")
        gemini_output_info = gr.Markdown(label="生成情報")

        with gr.Accordion("JSONログ", open=False):
            gemini_output_json = gr.Code(language="json", label="メタデータ")

        return gemini_output_image, gemini_output_info, gemini_output_json

    def _bind_events(self, controls: GeminiControls) -> None:
        # イベントハンドラ
        controls.template.change(
            fn=self.app.apply_template,
            inputs=[controls.template],
            outputs=[controls.prompt],
        )

        # Button 1: No optimization
        controls.gen_button_no_opt.click(
            fn=self.generate_gemini_image_no_optimization,
            inputs=[
                controls.prompt,
                controls.model,
                controls.aspect_ratio,
                controls.image_size,
                controls.google_search,
            ],
            outputs=[
                controls.output_image,
                controls.output_info,
                controls.output_json,
                controls.optimized_prompt,
            ],
        )

        # Button 2: Use pre-optimized
        controls.gen_button_use_opt.click(
            fn=self.generate_gemini_image_with_preoptimized,
            inputs=[
                controls.prompt,
                controls.optimized_prompt,
                controls.model,
                controls.aspect_ratio,
                controls.image_size,
                controls.google_search,
            ],
            outputs=[
                controls.output_image,
                controls.output_info,
                controls.output_json,
                controls.optimized_prompt,
            ],
        )

        # Button 3: Optimize then generate
        controls.gen_button_optimize_and_gen.click(
            fn=self.generate_gemini_image_with_optimization,
            inputs=[
                controls.prompt,
                controls.optimization_level,
                controls.model,
                controls.aspect_ratio,
                controls.image_size,
                controls.google_search,
            ],
            outputs=[
                controls.output_image,
                controls.output_info,
                controls.output_json,
                controls.optimized_prompt,
            ],
        )

        # Optimize prompt only button
        controls.optimize_prompt_button.click(
            fn=self.generate_optimized_prompt,
            inputs=[
                controls.prompt,
                controls.optimization_level,
            ],
            outputs=[
                controls.optimized_prompt,
            ],
        )

        controls.reset_button.click(
            fn=lambda: (
                "",
                "",
                1,
                self.app.default_aspect_ratio,
                DEFAULT_IMAGE_SIZE,
                False,
                None,
                "",
                "",
            ),
            inputs=[],
            outputs=[
                controls.prompt,
                controls.optimized_prompt,
                controls.optimization_level,
                controls.aspect_ratio,
                controls.image_size,
                controls.google_search,
                controls.output_image,
                controls.output_info,
                controls.output_json,
            ],
        )

    def _optimize_prompt_internal(
        self,
        prompt_text: str,
        optimization_level: int,
    ) -> Tuple[str, Optional[str]]:
        """
        共通のプロンプト最適化ヘルパー。

        Returns:
            (final_prompt, error_message or None)
        """
        if optimization_level <= 0:
            return prompt_text, None

        try:
            optimizer = PromptOptimizer(self.app.google_api_key)
            optimized_prompt, opt_error = optimizer.optimize(
                "",
                prompt_text,
                optimization_level,
            )
            if opt_error:
                logger.warning(
                    "event=gemini_prompt_optimize_warning tab=%s level=%s detail=%s",
                    TAB_GEMINI.key,
                    optimization_level,
                    opt_error,
                )
                return optimized_prompt, opt_error

            logger.info(
                "event=gemini_prompt_optimized tab=%s level=%s",
                TAB_GEMINI.key,
                optimization_level,
            )
            return optimized_prompt, None

        except Exception as e:
            logger.error(
                "event=gemini_prompt_optimize_error tab=%s level=%s error=%s",
                TAB_GEMINI.key,
                optimization_level,
                e,
                exc_info=True,
            )
            return prompt_text, str(e)

    def generate_optimized_prompt(
        self,
        prompt_text: str,
        optimization_level: int,
    ) -> str:
        """
        プロンプトのみを生成（画像生成なし）

        Args:
            prompt_text: ユーザーのプロンプト
            optimization_level: プロンプト最適化レベル（0/1/2）

        Returns:
            最適化されたプロンプト文字列
        """
        if not prompt_text or prompt_text.strip() == "":
            return "⚠️ プロンプトを入力してください"

        final_prompt, opt_error = self._optimize_prompt_internal(
            prompt_text,
            optimization_level,
        )
        if opt_error:
            return (
                f"⚠️ 最適化エラー: {opt_error}\n\nフォールバックプロンプト:\n{final_prompt}"
            )

        return final_prompt

    def generate_gemini_image(
        self,
        prompt: str,
        optimization_level: int,
        optimized_prompt_from_ui: str,
        model_name: str,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool,  # Phase 3.0
    ) -> Tuple[Optional[Image.Image], str, str, str]:
        """
        Geminiで画像を生成（Tab 1: Gemini専用）

        Args:
            prompt: ユーザーの元のプロンプト
            optimization_level: プロンプト最適化レベル（0/1/2）
            optimized_prompt_from_ui: UI経由で既に最適化されたプロンプト（空文字列なら新規生成）
            model_name: モデル名
            aspect_ratio: アスペクト比
            image_size: 画像サイズ
            enable_google_search: Google検索を有効化

        Returns:
            (image, info_text, json_log, optimized_prompt): 生成された画像、情報テキスト、JSONログ、最適化プロンプト
        """
        start_time = time.perf_counter()
        logger.info(
            "event=gemini_image_request tab=%s model=%s search=%s optimization_level=%s",
            TAB_GEMINI.key,
            model_name,
            enable_google_search,
            optimization_level,
        )

        if self.app.test_mode:
            return None, "⚠ テストモード: 画像生成機能は無効です", "", ""

        # 認証チェック
        if not self.app.google_api_key or self.app.gemini_generator is None:
            error_text = ERROR_API_KEY_NOT_CONFIGURED
            logger.error(
                "event=gemini_image_error tab=%s model=%s error=%s",
                TAB_GEMINI.key,
                model_name,
                "api_key_not_configured",
            )
            return None, error_text, "", ""

        try:
            # プロンプト選択ロジック
            optimized_prompt_info = None

            if optimized_prompt_from_ui and optimized_prompt_from_ui.strip():
                # UI経由で既に最適化されたプロンプトが渡された場合
                final_prompt = optimized_prompt_from_ui
                logger.info(
                    "event=gemini_prompt_source tab=%s source=ui_preoptimized",
                    TAB_GEMINI.key,
                )
            else:
                # プロンプト最適化
                final_prompt, opt_error = self._optimize_prompt_internal(
                    prompt,
                    optimization_level,
                )
                if opt_error:
                    optimized_prompt_info = opt_error

            # Gemini固有の設定（Phase 3.0: enable_google_search追加）
            config = GenerationConfig(
                model_type=ModelType.TEST
                if model_name == TEST_MODEL_NAME
                else ModelType.GEMINI,
                model_name=model_name,
                prompt=final_prompt,
                aspect_ratio=aspect_ratio,
                number_of_images=1,  # Geminiは常に1枚
                image_size=image_size,
                enable_google_search=enable_google_search,  # Phase 3.0
            )

            # 画像生成と後処理
            generator = self._select_generator(model_name)
            image_data_list, metadata = generator.generate(config)
            pil_image, info_text, json_log = self._build_image_outputs(
                model_name=model_name,
                image_data=image_data_list[0],
                metadata=metadata,
                optimization_level=optimization_level,
                optimized_prompt_info=optimized_prompt_info,
            )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "event=gemini_image_complete tab=%s model=%s search=%s optimization_level=%s duration_ms=%d",
                TAB_GEMINI.key,
                model_name,
                enable_google_search,
                optimization_level,
                duration_ms,
            )
            return pil_image, info_text, json_log, final_prompt

        except Exception as e:
            logger.error(
                "event=gemini_image_error tab=%s model=%s error=%s",
                TAB_GEMINI.key,
                model_name,
                e,
                exc_info=True,
            )
            error_text = f"❌ エラー: {str(e)}"
            return None, error_text, "", ""

    def _select_generator(self, model_name: str):
        if model_name == TEST_MODEL_NAME:
            return self.app.test_generator
        return self.app.gemini_generator

    def _build_image_outputs(
        self,
        *,
        model_name: str,
        image_data: bytes,
        metadata: dict,
        optimization_level: int,
        optimized_prompt_info: Optional[str],
    ) -> Tuple[Image.Image, str, str]:
        # PIL Imageに変換
        pil_image = Image.open(io.BytesIO(image_data))

        # ファイル保存（HF Spacesでは無効化される可能性あり）
        save_result = self.app.output_manager.save_image_with_metadata(
            image_data=image_data,
            metadata=metadata,
            prefix=GEMINI_OUTPUT_PREFIX,
            extension="jpg",
        )

        # 情報テキスト生成
        info_text = f"### 生成完了 ✅\n\n"
        info_text += f"**モデル**: {model_name}\n"
        if save_result:
            image_path, metadata_path = save_result
            info_text += f"**画像ファイル**: `{image_path.name}`\n"
            info_text += f"**メタデータ**: `{metadata_path.name}`\n"
        else:
            info_text += f"**画像ファイル**: （保存無効）\n"
        info_text += f"**ファイルサイズ**: {len(image_data) / 1024:.1f} KB\n"
        if optimization_level > 0:
            info_text += f"**プロンプト最適化**: レベル{optimization_level}\n"
        if optimized_prompt_info:
            info_text += f"**⚠️ 最適化警告**: {optimized_prompt_info}\n"

        # JSONログ
        json_log = json.dumps(metadata, ensure_ascii=False, indent=2)

        return pil_image, info_text, json_log

    def generate_gemini_image_no_optimization(
        self,
        prompt: str,
        model_name: str,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool,
    ) -> Tuple[Optional[Image.Image], str, str, str]:
        """
        Button 1: Generate image WITHOUT optimization (Level 0)

        ユーザーの元のプロンプトをそのまま使用して画像を生成します。
        最適化は行いません（内部的には optimization_level=0 を使用）。

        Args:
            prompt: ユーザーのプロンプト
            model_name: モデル名
            aspect_ratio: アスペクト比
            image_size: 画像サイズ
            enable_google_search: Google検索を有効化

        Returns:
            (image, info_text, json_log, optimized_prompt): 生成結果
        """
        logger.info(
            "event=gemini_button_click tab=%s button=1 mode=no_optimization",
            TAB_GEMINI.key,
        )

        return self.generate_gemini_image(
            prompt=prompt,
            optimization_level=0,
            optimized_prompt_from_ui="",
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            enable_google_search=enable_google_search,
        )

    def generate_gemini_image_with_preoptimized(
        self,
        original_prompt: str,
        optimized_prompt: str,
        model_name: str,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool,
    ) -> Tuple[Optional[Image.Image], str, str, str]:
        """
        Button 2: Generate image using pre-optimized prompt from textbox

        最適化プロンプト欄に表示されているテキストを使用して画像を生成します。
        追加の最適化は行いません。

        Args:
            original_prompt: ユーザーの元のプロンプト（フォールバック用）
            optimized_prompt: 最適化済みプロンプト（UI経由）
            model_name: モデル名
            aspect_ratio: アスペクト比
            image_size: 画像サイズ
            enable_google_search: Google検索を有効化

        Returns:
            (image, info_text, json_log, optimized_prompt): 生成結果
        """
        logger.info(
            "event=gemini_button_click tab=%s button=2 mode=pre_optimized",
            TAB_GEMINI.key,
        )

        # Error handling: Check if optimized prompt exists
        if not optimized_prompt or optimized_prompt.strip() == "":
            logger.warning(
                "event=gemini_button_warning tab=%s button=2 reason=no_optimized_prompt",
                TAB_GEMINI.key,
            )
            error_text = WARNING_EMPTY_OPTIMIZED_PROMPT
            # Fallback to original prompt with no optimization
            return self.generate_gemini_image(
                prompt=original_prompt,
                optimization_level=0,
                optimized_prompt_from_ui="",
                model_name=model_name,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                enable_google_search=enable_google_search,
            )

        # Check for error markers in optimized prompt
        if optimized_prompt.startswith("⚠️"):
            logger.warning(
                "event=gemini_button_warning tab=%s button=2 reason=optimized_prompt_has_error",
                TAB_GEMINI.key,
            )
            error_text = ERROR_OPTIMIZED_PROMPT_HAS_ERROR
            return None, error_text, "", optimized_prompt

        # Use pre-optimized prompt
        return self.generate_gemini_image(
            prompt=original_prompt,
            optimization_level=0,
            optimized_prompt_from_ui=optimized_prompt,
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            enable_google_search=enable_google_search,
        )

    def generate_gemini_image_with_optimization(
        self,
        prompt: str,
        optimization_level: int,
        model_name: str,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool,
    ) -> Tuple[Optional[Image.Image], str, str, str]:
        """
        Button 3: Optimize prompt first, then generate image

        Radioボタンで選択されたレベル（1-3）でプロンプトを最適化してから画像を生成します。
        最適化されたプロンプトは最適化プロンプト欄に表示されます。

        Args:
            prompt: ユーザーのプロンプト
            optimization_level: 最適化レベル（1-3）
            model_name: モデル名
            aspect_ratio: アスペクト比
            image_size: 画像サイズ
            enable_google_search: Google検索を有効化

        Returns:
            (image, info_text, json_log, optimized_prompt): 生成結果
        """
        logger.info(
            "event=gemini_button_click tab=%s button=3 mode=optimize_and_generate level=%s",
            TAB_GEMINI.key,
            optimization_level,
        )

        return self.generate_gemini_image(
            prompt=prompt,
            optimization_level=optimization_level,
            optimized_prompt_from_ui="",
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            enable_google_search=enable_google_search,
        )
