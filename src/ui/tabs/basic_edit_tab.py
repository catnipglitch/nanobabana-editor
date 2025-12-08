"""
Basic Edit Tab

ベーシック編集タブ（シンプルな1回編集・アップスケール）
Tab 2: 入力画像1枚 + プロンプト（テンプレート/最適化）による画像生成
"""

import gradio as gr
import logging
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from .base_tab import BaseTab
from ...core.generators import GenerationConfig, ModelType
from ...core.tab_specs import TAB_BASIC_EDIT
from ...core.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


@dataclass
class BasicEditControls:
    model: Any
    template: Any
    prompt: Any
    optimized_prompt: Any
    optimization_level: Any
    optimize_prompt_button: Any
    input_image: Any  # NEW: Single input image
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
GEMINI_OUTPUT_PREFIX = "gemini_edit"
DEFAULT_IMAGE_SIZE = "1K"

ERROR_API_KEY_NOT_CONFIGURED = """❌ エラー: APIキーが設定されていません

**Settings タブ** でGoogle API Keyを設定してください。
"""

WARNING_EMPTY_OPTIMIZED_PROMPT = """⚠️ 警告: 最適化済みプロンプトが空です

元のプロンプトを使用して生成します。
先に「画像生成（プロンプトの最適化を実施してから生成）」を実行してください。
"""

ERROR_OPTIMIZED_PROMPT_HAS_ERROR = """⚠️ エラー: 最適化済みプロンプトにエラーが含まれています

「画像生成（プロンプトの最適化を実施してから生成）」を再実行してください。
"""


class BasicEditTab(BaseTab):
    """Basic Edit (Image-to-Image) Tab"""

    def create_ui(self) -> None:
        """Create Basic Edit tab UI"""
        with gr.Tab(
            TAB_BASIC_EDIT.display_name,
            id=TAB_BASIC_EDIT.key,
            elem_id=TAB_BASIC_EDIT.elem_id,
        ):
            controls = self._build_layout()
            self._bind_events(controls)

    def _build_layout(self) -> BasicEditControls:
        with gr.Row():
            with gr.Column(scale=1):
                (
                    edit_model,
                    edit_template,
                    edit_prompt,
                    edit_optimized_prompt,
                    edit_optimization_level,
                    edit_optimize_prompt_button,
                    edit_input_image,
                    edit_aspect_ratio,
                    edit_image_size,
                    edit_google_search,
                    edit_gen_button_optimize_and_gen,
                    edit_gen_button_use_opt,
                    edit_gen_button_no_opt,
                    edit_reset_button,
                ) = self._build_input_column()

            with gr.Column(scale=1):
                edit_output_image, edit_output_info, edit_output_json = (
                    self._build_output_column()
                )

        return BasicEditControls(
            model=edit_model,
            template=edit_template,
            prompt=edit_prompt,
            optimized_prompt=edit_optimized_prompt,
            optimization_level=edit_optimization_level,
            optimize_prompt_button=edit_optimize_prompt_button,
            input_image=edit_input_image,
            aspect_ratio=edit_aspect_ratio,
            image_size=edit_image_size,
            google_search=edit_google_search,
            gen_button_optimize_and_gen=edit_gen_button_optimize_and_gen,
            gen_button_use_opt=edit_gen_button_use_opt,
            gen_button_no_opt=edit_gen_button_no_opt,
            reset_button=edit_reset_button,
            output_image=edit_output_image,
            output_info=edit_output_info,
            output_json=edit_output_json,
        )

    def _build_input_column(self):
        # モデル選択
        default_model = "gemini-3-pro-image-preview"
        model_value = (
            default_model
            if default_model in self.app.gemini_models
            else (self.app.gemini_models[0] if self.app.gemini_models else None)
        )

        edit_model = gr.Dropdown(
            label="モデル",
            choices=self.app.gemini_models,
            value=model_value,
            info="Gemini画像生成モデル",
        )

        # プロンプトテンプレート (image_edit カテゴリ)
        edit_template = gr.Dropdown(
            label="プロンプトテンプレート",
            choices=self.app.template_manager.get_template_choices_for_tab(
                "gemini_edit01"  # Maps to 'image_edit' category
            ),
            value="選択してください",
            info="サンプルプロンプトを選択",
        )

        # ユーザープロンプト
        edit_prompt = gr.Textbox(
            label="ユーザープロンプト",
            placeholder="生成したい画像を説明してください",
            lines=4,
        )

        # 最適化されたプロンプト表示
        edit_optimized_prompt = gr.Textbox(
            label="最適化されたプロンプト",
            placeholder="「プロンプト生成」または「編集開始」をクリックすると、最適化されたプロンプトが表示されます",
            lines=5,
            interactive=True,
            info="生成前にプロンプトを確認・編集できます",
        )

        # プロンプト最適化レベル
        edit_optimization_level = gr.Radio(
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
        edit_optimize_prompt_button = gr.Button(
            "最適化プロンプト生成",
            variant="secondary",
            size="sm",
        )

        # 入力画像 (Single)
        # type="filepath" to get filename for optimizer context
        edit_input_image = gr.Image(
            label="入力画像",
            type="filepath",
            sources=["upload", "clipboard"],
            height=300,
        )

        # 詳細設定
        with gr.Accordion("詳細設定", open=True):
            edit_aspect_ratio = gr.Radio(
                label="アスペクト比",
                choices=[
                    "1:1", "2:3", "3:2", "3:4", "4:3",
                    "4:5", "5:4", "9:16", "16:9", "21:9"
                ],
                value=self.app.default_aspect_ratio,
                info="生成画像の縦横比（全10種類サポート）",
            )

            edit_image_size = gr.Radio(
                label="解像度",
                choices=["1K", "2K", "4K"],
                value=DEFAULT_IMAGE_SIZE,
                info="Gemini 3 Pro Image Preview で有効（他モデルでは無視されます）",
            )

        # ツールオプション
        with gr.Accordion("ツールオプション", open=False):
            edit_google_search = gr.Checkbox(
                label="Google Search",
                value=False,
                info="Google検索でリアルタイム情報を取得（Gemini 3 Pro Image推奨）"
            )

        # ボタン（2行2列レイアウト）
        with gr.Row():
            edit_gen_button_optimize_and_gen = gr.Button(
                "画像生成（プロンプトの最適化を実施してから生成）",
                variant="primary",
                size="sm",
            )
            edit_gen_button_use_opt = gr.Button(
                "画像生成（最適化済みプロンプトで生成）",
                variant="secondary",
                size="sm",
            )

        with gr.Row():
            edit_gen_button_no_opt = gr.Button(
                "画像生成（ユーザープロンプトで生成）",
                variant="secondary",
                size="sm",
            )
            edit_reset_button = gr.Button(
                "リセット",
                size="sm",
            )

        return (
            edit_model,
            edit_template,
            edit_prompt,
            edit_optimized_prompt,
            edit_optimization_level,
            edit_optimize_prompt_button,
            edit_input_image,
            edit_aspect_ratio,
            edit_image_size,
            edit_google_search,
            edit_gen_button_optimize_and_gen,
            edit_gen_button_use_opt,
            edit_gen_button_no_opt,
            edit_reset_button,
        )

    def _build_output_column(self):
        edit_output_image = gr.Image(label="生成された画像", type="pil")
        edit_output_info = gr.Markdown(label="生成情報")
        with gr.Accordion("JSONログ", open=False):
            edit_output_json = gr.Code(language="json", label="メタデータ")
        return edit_output_image, edit_output_info, edit_output_json

    def _bind_events(self, controls: BasicEditControls) -> None:
        # テンプレート適用
        controls.template.change(
            fn=self.app.apply_template,
            inputs=[controls.template],
            outputs=[controls.prompt],
        )

        # Button 1: No optimization
        controls.gen_button_no_opt.click(
            fn=self.generate_image_no_optimization,
            inputs=[
                controls.prompt,
                controls.input_image,
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
            fn=self.generate_image_with_preoptimized,
            inputs=[
                controls.prompt,
                controls.optimized_prompt,
                controls.input_image,
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
            fn=self.generate_image_with_optimization,
            inputs=[
                controls.prompt,
                controls.optimization_level,
                controls.input_image,
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

        # Optimize prompt only
        controls.optimize_prompt_button.click(
            fn=self.generate_optimized_prompt,
            inputs=[
                controls.prompt,
                controls.optimization_level,
                controls.input_image,
            ],
            outputs=[
                controls.optimized_prompt,
            ],
        )

        # Reset
        controls.reset_button.click(
            fn=lambda: (
                "", "", 1, None, self.app.default_aspect_ratio, DEFAULT_IMAGE_SIZE, False, None, "", ""
            ),
            outputs=[
                controls.prompt,
                controls.optimized_prompt,
                controls.optimization_level,
                controls.input_image,
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
        image_path: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """
        共通のプロンプト最適化ヘルパー
        """
        if optimization_level <= 0:
            return prompt_text, None

        file_paths = [image_path] if image_path else None

        try:
            optimizer = PromptOptimizer(self.app.google_api_key)
            # Tab2では Base Prompt が空の場合、Input Prompt自体をBaseとして扱う
            # またはテンプレート選択によって prompt_text に既に Base Prompt が入っている
            optimized_prompt, opt_error = optimizer.optimize(
                base_prompt="",  # Base prompt is integrated in prompt_text
                user_instructions=prompt_text,
                level=optimization_level,
                file_paths=file_paths,
            )
            if opt_error:
                return optimized_prompt, opt_error
            return optimized_prompt, None

        except Exception as e:
            logger.error(f"Optimization error: {e}", exc_info=True)
            return prompt_text, str(e)

    def generate_optimized_prompt(
        self,
        prompt_text: str,
        optimization_level: int,
        image_path: Optional[str],
    ) -> str:
        """プロンプトのみ生成"""
        if not prompt_text.strip() and not image_path:
            return "⚠️ プロンプトを入力するか、画像を選択してください"

        final_prompt, opt_error = self._optimize_prompt_internal(
            prompt_text, optimization_level, image_path
        )
        if opt_error:
            return f"⚠️ 最適化エラー: {opt_error}\n\nフォールバック:\n{final_prompt}"
        return final_prompt

    def _generate_image_core(
        self,
        prompt: str,
        optimization_level: int,
        optimized_prompt_from_ui: str,
        input_image_path: Optional[str],
        model_name: str,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool,
    ) -> Tuple[Optional[Image.Image], str, str, str]:
        """
        画像生成の共通ロジック
        """
        start_time = time.perf_counter()

        if self.app.test_mode:
            return None, "⚠ テストモード: 画像生成機能は無効です", "", ""

        if not self.app.google_api_key or self.app.gemini_generator is None:
            return None, ERROR_API_KEY_NOT_CONFIGURED, "", ""

        if not input_image_path:
            return None, "⚠️ エラー: 入力画像を選択してください", "", ""

        try:
            # プロンプト決定
            optimized_prompt_info = None
            if optimized_prompt_from_ui and optimized_prompt_from_ui.strip():
                final_prompt = optimized_prompt_from_ui
            else:
                final_prompt, opt_error = self._optimize_prompt_internal(
                    prompt, optimization_level, input_image_path
                )
                if opt_error:
                    optimized_prompt_info = opt_error

            # 画像読み込み
            try:
                input_pil_image = Image.open(input_image_path)
            except Exception as e:
                return None, f"❌ 画像読み込みエラー: {e}", "", ""

            # 設定
            config = GenerationConfig(
                model_type=ModelType.TEST if model_name == TEST_MODEL_NAME else ModelType.GEMINI,
                model_name=model_name,
                prompt=final_prompt,
                aspect_ratio=aspect_ratio,
                number_of_images=1,
                image_size=image_size,
                enable_google_search=enable_google_search,
                reference_images=[input_pil_image], # Single input image as reference
            )

            # 生成実行
            generator = self.app.test_generator if model_name == TEST_MODEL_NAME else self.app.gemini_generator
            image_data_list, metadata = generator.generate(config)

            if not image_data_list:
                return None, "❌ 生成された画像データがありません", "", final_prompt

            # 結果構築
            pil_image = Image.open(io.BytesIO(image_data_list[0]))
            
            save_result = self.app.output_manager.save_image_with_metadata(
                image_data=image_data_list[0],
                metadata=metadata,
                prefix=GEMINI_OUTPUT_PREFIX,
                extension="jpg",
            )

            info_text = f"### 生成完了 ✅\n\n**モデル**: {model_name}\n"
            if save_result:
                image_path, metadata_path = save_result
                info_text += f"**画像ファイル**: `{image_path.name}`\n"
                info_text += f"**メタデータ**: `{metadata_path.name}`\n"
            else:
                info_text += "**ファイル保存**: 無効\n"
            
            if optimized_prompt_info:
                info_text += f"**⚠️ 最適化警告**: {optimized_prompt_info}\n"

            json_log = json.dumps(metadata, ensure_ascii=False, indent=2)

            return pil_image, info_text, json_log, final_prompt

        except Exception as e:
            logger.error(f"Image generation error: {e}", exc_info=True)
            return None, f"❌ エラー: {str(e)}", "", ""

    def generate_image_no_optimization(
        self,
        prompt,
        input_image,
        model,
        ar,
        size,
        search
    ):
        return self._generate_image_core(
            prompt, 0, "", input_image, model, ar, size, search
        )

    def generate_image_with_preoptimized(
        self,
        prompt, opt_prompt, input_image, model, ar, size, search
    ):
        if not opt_prompt.strip():
            return self.generate_image_no_optimization(prompt, input_image, model, ar, size, search)
        return self._generate_image_core(
            prompt, 0, opt_prompt, input_image, model, ar, size, search
        )

    def generate_image_with_optimization(
        self,
        prompt, level, input_image, model, ar, size, search
    ):
        return self._generate_image_core(
            prompt, level, "", input_image, model, ar, size, search
        )