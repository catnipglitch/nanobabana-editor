"""
Reference Tab

Reference image-based generation tab (Gemini with reference images).
"""

import io
import logging

import gradio as gr
from PIL import Image
from .base_tab import BaseTab
from ...core.generators import GenerationConfig, ModelType
from ...core.tab_specs import TAB_REFERENCE

logger = logging.getLogger(__name__)


class ReferenceTab(BaseTab):
    """Reference image-based generation tab"""

    def create_ui(self) -> None:
        """Create Reference tab UI"""
        with gr.Tab(TAB_REFERENCE.display_name, id=TAB_REFERENCE.key, elem_id=TAB_REFERENCE.elem_id):
            gr.Markdown("""
            ### Gemini による参照画像ベース画像生成

            最大14枚の参照画像と共にプロンプトを入力して、新しい画像を生成します。
            参照画像なしでも生成可能です（text-to-image）。
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    edit_model = gr.Dropdown(
                        label="モデル",
                        choices=[
                            "gemini-2.5-flash-image",
                            "gemini-3-pro-image-preview"
                        ],
                        value="gemini-3-pro-image-preview",
                        info="Gemini画像生成モデル"
                    )

                    # プロンプトテンプレート選択
                    edit_template = gr.Dropdown(
                        label="プロンプトテンプレート",
                        choices=self.app.template_manager.get_template_choices_for_tab("gemini_edit02"),
                        value="選択してください",
                        info="サンプルプロンプトを選択"
                    )

                    edit_prompt = gr.Textbox(
                        label="ユーザープロンプト",
                        placeholder="生成したい画像を説明してください",
                        lines=4
                    )

                    # 最適化されたプロンプト表示
                    optimized_prompt_display = gr.Textbox(
                        label="最適化されたプロンプト",
                        placeholder="「プロンプト生成」または「編集開始」をクリックすると、最適化されたプロンプトが表示されます",
                        lines=5,
                        interactive=True,
                        info="生成前にプロンプトを確認・編集できます"
                    )

                    # プロンプト最適化レベル（Tab 1 と同様に 1-3 のみ表示）
                    optimization_level = gr.Radio(
                        label="プロンプト最適化レベル",
                        choices=[
                            ("レベル1: 修正のみ（誤字脱字・語順）", 1),
                            ("レベル2: 標準最適化（推奨）", 2),
                            ("レベル3: 創造的拡張", 3),
                        ],
                        value=1,
                        info="レベル1推奨: Gemini 2.0 Flashで誤字脱字を修正"
                    )

                    # 最適化プロンプト生成ボタン（Tab 1 と揃えた位置）
                    generate_prompt_button = gr.Button(
                        "最適化プロンプト生成",
                        variant="secondary",
                        size="sm",
                    )

                    # 参照画像アップロード（段階的表示）
                    # 標準表示: 参照画像 1-3
                    ref_image_1 = gr.Image(label="参照画像 1", type="filepath", sources=["upload", "clipboard"])
                    ref_image_2 = gr.Image(label="参照画像 2", type="filepath", sources=["upload", "clipboard"])
                    ref_image_3 = gr.Image(label="参照画像 3", type="filepath", sources=["upload", "clipboard"])

                    # アコーディオン1: 参照画像 4-6（デフォルトで閉じる）
                    with gr.Accordion("参照画像 4-6（追加）", open=False):
                        ref_image_4 = gr.Image(label="参照画像 4", type="filepath", sources=["upload", "clipboard"])
                        ref_image_5 = gr.Image(label="参照画像 5", type="filepath", sources=["upload", "clipboard"])
                        ref_image_6 = gr.Image(label="参照画像 6", type="filepath", sources=["upload", "clipboard"])

                    # アコーディオン2: 参照画像 7-14（デフォルトで閉じる）
                    with gr.Accordion("参照画像 7-14（さらに追加）", open=False):
                        ref_image_7 = gr.Image(label="参照画像 7", type="filepath", sources=["upload", "clipboard"])
                        ref_image_8 = gr.Image(label="参照画像 8", type="filepath", sources=["upload", "clipboard"])
                        ref_image_9 = gr.Image(label="参照画像 9", type="filepath", sources=["upload", "clipboard"])
                        ref_image_10 = gr.Image(label="参照画像 10", type="filepath", sources=["upload", "clipboard"])
                        ref_image_11 = gr.Image(label="参照画像 11", type="filepath", sources=["upload", "clipboard"])
                        ref_image_12 = gr.Image(label="参照画像 12", type="filepath", sources=["upload", "clipboard"])
                        ref_image_13 = gr.Image(label="参照画像 13", type="filepath", sources=["upload", "clipboard"])
                        ref_image_14 = gr.Image(label="参照画像 14", type="filepath", sources=["upload", "clipboard"])

                    # 画像カウント表示
                    edit_image_count = gr.Markdown("**アップロード**: 0/14 枚")

                    # 詳細設定
                    with gr.Accordion("詳細設定", open=True):
                        edit_aspect_ratio = gr.Radio(
                            label="アスペクト比",
                            choices=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            value="1:1",
                            info="生成画像の縦横比（全10種類サポート）"
                        )
                        edit_image_size = gr.Radio(
                            label="解像度",
                            choices=["1K", "2K", "4K"],
                            value="1K",
                            info="Gemini 3 Pro Image Preview で有効（他モデルでは無視されます）"
                        )

                    # ツールオプション
                    with gr.Accordion("ツールオプション", open=False):
                        enable_google_search = gr.Checkbox(
                            label="Google Search",
                            value=False,
                            info="Google検索でリアルタイム情報を取得（Gemini 3 Pro Image推奨）"
                        )

                    # 画像生成ボタン（Tab 1 と同じ構成）
                    with gr.Row():
                        reference_gen_button_optimize_and_gen = gr.Button(
                            "画像生成（プロンプトの最適化を実施してから生成）",
                            variant="primary",
                            size="sm",
                        )
                        reference_gen_button_use_opt = gr.Button(
                            "画像生成（最適化済みプロンプトで生成）",
                            variant="secondary",
                            size="sm",
                        )

                    with gr.Row():
                        reference_gen_button_no_opt = gr.Button(
                            "画像生成（ユーザープロンプトで生成）",
                            variant="secondary",
                            size="sm",
                        )
                        reset_button = gr.Button(
                            "リセット",
                            size="sm",
                        )

                with gr.Column(scale=1):
                    edit_output_image = gr.Image(label="生成された画像", type="pil")
                    edit_output_info = gr.Markdown(label="生成情報")
                    edit_text_response = gr.Textbox(
                        label="テキスト応答（Geminiからの説明）",
                        lines=5,
                        interactive=False
                    )

            # 全14個のImageコンポーネントをリストに格納
            ref_images = [
                ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
                ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
                ref_image_11, ref_image_12, ref_image_13, ref_image_14
            ]

            # 各Imageコンポーネントのchange イベントでカウント更新
            for ref_img in ref_images:
                ref_img.change(
                    fn=self.update_image_count,
                    inputs=ref_images,  # 全14個を入力
                    outputs=[edit_image_count]
                )

            # テンプレート選択時の処理
            edit_template.change(
                fn=self.app.apply_template,
                inputs=[edit_template],
                outputs=[edit_prompt]
            )

            # プロンプト生成ボタン
            generate_prompt_button.click(
                fn=self.generate_optimized_prompt,
                inputs=[edit_prompt, optimization_level] + ref_images,
                outputs=[optimized_prompt_display]
            )

            # 画像生成ボタン（Tab 1 と同等の3パターン）
            # 1) プロンプトを最適化してから生成
            reference_gen_button_optimize_and_gen.click(
                fn=self.generate_reference_image_with_optimization,
                inputs=[edit_model, edit_prompt, optimized_prompt_display, optimization_level]
                + ref_images
                + [edit_aspect_ratio, edit_image_size, enable_google_search],
                outputs=[
                    edit_output_image,
                    edit_output_info,
                    edit_text_response,
                    optimized_prompt_display,
                ],
            )

            # 2) 最適化済みプロンプトで生成
            reference_gen_button_use_opt.click(
                fn=self.generate_reference_image_with_preoptimized,
                inputs=[edit_model, edit_prompt, optimized_prompt_display, optimization_level]
                + ref_images
                + [edit_aspect_ratio, edit_image_size, enable_google_search],
                outputs=[
                    edit_output_image,
                    edit_output_info,
                    edit_text_response,
                    optimized_prompt_display,
                ],
            )

            # 3) ユーザープロンプトでそのまま生成（最適化なし）
            reference_gen_button_no_opt.click(
                fn=self.generate_reference_image_no_optimization,
                inputs=[edit_model, edit_prompt, optimized_prompt_display, optimization_level]
                + ref_images
                + [edit_aspect_ratio, edit_image_size, enable_google_search],
                outputs=[
                    edit_output_image,
                    edit_output_info,
                    edit_text_response,
                    optimized_prompt_display,
                ],
            )

            # リセットボタン
            reset_button.click(
                fn=lambda: (
                    None, None, None, None, None, None, None, None, None, None, None, None, None, None,  # 14 ref images
                    "",  # edit_prompt
                    "",  # optimized_prompt_display
                    "1:1",  # edit_aspect_ratio
                    "1K",  # edit_image_size
                    False,  # enable_google_search
                    None,  # edit_output_image
                    "",  # edit_output_info
                    ""  # edit_text_response
                ),
                inputs=[],
                outputs=ref_images + [edit_prompt, optimized_prompt_display, edit_aspect_ratio, edit_image_size, enable_google_search, edit_output_image, edit_output_info, edit_text_response]
            )

    def update_image_count(self, *images):
        """
        アップロードされた画像数をカウント（Accordion版）

        Args:
            *images: 14個のgr.Image値（None or filepath）

        Returns:
            count_text: 画像数テキスト
        """
        # Noneでない画像をカウント
        count = sum(1 for img in images if img is not None)
        return f"**アップロード**: {count}/14 枚"

    def generate_optimized_prompt(
        self,
        prompt: str,
        optimization_level: int,
        ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
        ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
        ref_image_11, ref_image_12, ref_image_13, ref_image_14,
    ) -> str:
        """
        プロンプトのみを生成（画像生成なし）

        プレビュー専用メソッド。最適化されたプロンプトを生成して返す。
        ファイル名情報はPromptOptimizer内部で自動追記される。

        Args:
            prompt: ユーザーの入力プロンプト
            optimization_level: プロンプト最適化レベル（0/1/2/3）
            ref_image_1-14: 参照画像のパス（None or filepath）

        Returns:
            最適化されたプロンプト文字列
        """
        from ...core.prompt_optimizer import PromptOptimizer

        # 14個の参照画像をリスト化
        ref_images = [
            ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
            ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
            ref_image_11, ref_image_12, ref_image_13, ref_image_14
        ]

        # Noneを除外してfile_pathsを取得
        file_paths = [img for img in ref_images if img is not None]

        # 基本プロンプトは空（参照画像ベースなので）
        base_prompt = ""

        # プロンプト最適化（ファイルリストはOptimizer内部で自動追加）
        try:
            optimizer = PromptOptimizer(self.app.google_api_key)

            # 全レベルでoptimize()を使用し、file_pathsを渡す
            optimized_prompt, opt_error = optimizer.optimize(
                base_prompt, prompt, optimization_level, file_paths=file_paths
            )

            if opt_error:
                return f"⚠️ 最適化エラー: {opt_error}\n\nフォールバックプロンプト:\n{optimized_prompt}"

            return optimized_prompt

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}", exc_info=True)
            # フォールバック: Level 0で再試行
            try:
                optimizer = PromptOptimizer(self.app.google_api_key)
                fallback_prompt, _ = optimizer.optimize(
                    base_prompt, prompt, 0, file_paths=file_paths
                )
                return f"⚠️ 最適化エラー: {str(e)}\n\nフォールバックプロンプト:\n{fallback_prompt}"
            except Exception:
                return f"⚠️ エラー: {str(e)}\n\n元のプロンプト:\n{prompt}"

    def generate_reference_image_no_optimization(
        self,
        model_name: str,
        prompt: str,
        optimized_prompt_display: str,
        optimization_level: int,
        ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
        ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
        ref_image_11, ref_image_12, ref_image_13, ref_image_14,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool = False,
    ):
        """Button 3: ユーザープロンプトでそのまま生成（最適化なし）"""
        return self.generate_with_reference_images(
            model_name=model_name,
            prompt=prompt,
            optimized_prompt_display="",
            optimization_level=0,
            ref_image_1=ref_image_1,
            ref_image_2=ref_image_2,
            ref_image_3=ref_image_3,
            ref_image_4=ref_image_4,
            ref_image_5=ref_image_5,
            ref_image_6=ref_image_6,
            ref_image_7=ref_image_7,
            ref_image_8=ref_image_8,
            ref_image_9=ref_image_9,
            ref_image_10=ref_image_10,
            ref_image_11=ref_image_11,
            ref_image_12=ref_image_12,
            ref_image_13=ref_image_13,
            ref_image_14=ref_image_14,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            enable_google_search=enable_google_search,
        )

    def generate_reference_image_with_preoptimized(
        self,
        model_name: str,
        prompt: str,
        optimized_prompt_display: str,
        optimization_level: int,
        ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
        ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
        ref_image_11, ref_image_12, ref_image_13, ref_image_14,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool = False,
    ):
        """Button 2: 最適化済みプロンプトを使用して生成"""
        # 最適化済みプロンプトが空の場合は警告を返す
        if not optimized_prompt_display or optimized_prompt_display.strip() == "":
            warning_text = """⚠️ 警告: 最適化済みプロンプトが空です

元のプロンプトを使用して生成します。
先に「画像生成（プロンプトの最適化を実施してから生成）」を実行してください。
"""
            return None, warning_text, "", optimized_prompt_display

        # エラーマーカーが含まれている場合はそのままエラーとして扱う
        if optimized_prompt_display.startswith("⚠️"):
            error_text = """⚠️ エラー: 最適化済みプロンプトにエラーが含まれています

「画像生成（プロンプトの最適化を実施してから生成）」を再実行してください。
"""
            return None, error_text, "", optimized_prompt_display

        # 最適化済みプロンプトをそのまま使用して生成
        return self.generate_with_reference_images(
            model_name=model_name,
            prompt=prompt,
            optimized_prompt_display=optimized_prompt_display,
            optimization_level=0,
            ref_image_1=ref_image_1,
            ref_image_2=ref_image_2,
            ref_image_3=ref_image_3,
            ref_image_4=ref_image_4,
            ref_image_5=ref_image_5,
            ref_image_6=ref_image_6,
            ref_image_7=ref_image_7,
            ref_image_8=ref_image_8,
            ref_image_9=ref_image_9,
            ref_image_10=ref_image_10,
            ref_image_11=ref_image_11,
            ref_image_12=ref_image_12,
            ref_image_13=ref_image_13,
            ref_image_14=ref_image_14,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            enable_google_search=enable_google_search,
        )

    def generate_reference_image_with_optimization(
        self,
        model_name: str,
        prompt: str,
        optimized_prompt_display: str,
        optimization_level: int,
        ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
        ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
        ref_image_11, ref_image_12, ref_image_13, ref_image_14,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool = False,
    ):
        """Button 1: プロンプトを最適化してから生成"""
        return self.generate_with_reference_images(
            model_name=model_name,
            prompt=prompt,
            optimized_prompt_display="",
            optimization_level=optimization_level,
            ref_image_1=ref_image_1,
            ref_image_2=ref_image_2,
            ref_image_3=ref_image_3,
            ref_image_4=ref_image_4,
            ref_image_5=ref_image_5,
            ref_image_6=ref_image_6,
            ref_image_7=ref_image_7,
            ref_image_8=ref_image_8,
            ref_image_9=ref_image_9,
            ref_image_10=ref_image_10,
            ref_image_11=ref_image_11,
            ref_image_12=ref_image_12,
            ref_image_13=ref_image_13,
            ref_image_14=ref_image_14,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            enable_google_search=enable_google_search,
        )

    def generate_with_reference_images(
        self,
        model_name: str,
        prompt: str,
        optimized_prompt_display: str,
        optimization_level: int,
        ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
        ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
        ref_image_11, ref_image_12, ref_image_13, ref_image_14,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool = False
    ):
        """
        参照画像をベースに画像生成する（Tab 3用 - Phase 2.6、Accordion版 + プロンプト最適化対応）

        Args:
            model_name: 使用するモデル名
            prompt: ユーザー入力プロンプト
            optimized_prompt_display: 最適化されたプロンプト（空の場合は自動生成）
            optimization_level: プロンプト最適化レベル（0/1/2/3）
            ref_image_1-14: 参照画像のパス（None or filepath）
            aspect_ratio: アスペクト比
            image_size: 画像サイズ
            enable_google_search: Google検索を有効にするか（default: False）

        Returns:
            (generated_image, info_text, text_response, optimized_prompt): 生成画像、情報テキスト、テキスト応答、最適化プロンプト
        """
        # 14個の参照画像をリスト化
        ref_images = [
            ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
            ref_image_6, ref_image_7, ref_image_8, ref_image_9, ref_image_10,
            ref_image_11, ref_image_12, ref_image_13, ref_image_14
        ]

        # Noneを除外してfile_pathsを生成
        file_paths = [img for img in ref_images if img is not None]
        if not prompt or prompt.strip() == "":
            return None, "⚠ プロンプトを入力してください", "", ""

        # 認証チェック
        if not self.app.google_api_key or self.app.gemini_generator is None:
            error_text = """❌ エラー: APIキーが設定されていません

**Settings タブ** でGoogle API Keyを設定してください。

1. Settings タブを開く
2. APIキーを入力
3. 「接続テスト」ボタンで確認
4. 「APIキーを適用」ボタンで適用
"""
            logger.error("API key not configured")
            return None, error_text, "", ""

        try:
            # プロンプト最適化
            if optimized_prompt_display and optimized_prompt_display.strip():
                # 既に最適化されたプロンプトがある場合はそれを使用
                final_prompt = optimized_prompt_display.strip()
            else:
                # 最適化されたプロンプトがない場合は自動生成
                from ...core.prompt_optimizer import PromptOptimizer

                base_prompt = ""
                optimizer = PromptOptimizer(self.app.google_api_key)

                # file_pathsを渡してプロンプト最適化（ファイルリストは内部で自動追加）
                final_prompt, opt_error = optimizer.optimize(
                    base_prompt, prompt, optimization_level, file_paths=file_paths
                )
                if opt_error:
                    logger.warning(f"Prompt optimization warning: {opt_error}")

            # 参照画像の処理（最大14枚）- File component returns paths directly
            ref_images_list = None
            if file_paths and len(file_paths) > 0:
                ref_images_list = []
                for file_path in file_paths[:14]:  # 最大14枚に制限
                    try:
                        img = Image.open(file_path)
                        ref_images_list.append(img)
                    except Exception as e:
                        logger.error(f"Failed to load reference image {file_path}: {e}")

                if not ref_images_list:
                    ref_images_list = None
                else:
                    logger.info(f"Using {len(ref_images_list)} reference images")

            # GenerationConfig作成
            config = GenerationConfig(
                model_type=ModelType.GEMINI,
                model_name=model_name,
                prompt=final_prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                reference_images=ref_images_list,
                number_of_images=1,
                enable_google_search=enable_google_search,
            )

            # 画像生成
            image_data_list, metadata = self.app.gemini_generator.generate(config)

            if not image_data_list:
                return None, "❌ 画像の生成に失敗しました", "", final_prompt

            # 生成した画像をPIL Imageに変換
            generated_image = Image.open(io.BytesIO(image_data_list[0]))

            # 画像とメタデータを保存（HF Spacesでは無効化される可能性あり）
            save_result = self.app.output_manager.save_image_with_metadata(
                image_data=image_data_list[0],
                metadata=metadata,
                prefix="edit_gen",
                extension="jpg",
            )

            # テキスト応答を取得
            text_response = metadata.get("text_response", "")

            # 情報テキスト作成
            info_text = f"""✓ 画像生成完了!

**モデル**: {model_name}
**プロンプト**: {final_prompt[:100]}{'...' if len(final_prompt) > 100 else ''}
**アスペクト比**: {aspect_ratio}
**解像度**: {image_size}
**参照画像数**: {len(ref_images_list) if ref_images_list else 0}
"""
            if save_result:
                image_path, metadata_path = save_result
                info_text += f"**画像ファイル**: `{image_path.name}`\n"
                info_text += f"**メタデータ**: `{metadata_path.name}`\n"
                logger.info(f"Reference image generation complete: {image_path.name}")
            else:
                info_text += f"**画像ファイル**: （保存無効）\n"
                logger.info("Reference image generation complete (save disabled)")

            info_text += f"**サイズ**: {len(image_data_list[0]):,} バイト\n"
            return generated_image, info_text, text_response, final_prompt

        except Exception as e:
            logger.error(f"Reference image generation failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, error_text, "", ""
