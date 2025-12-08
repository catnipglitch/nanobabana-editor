"""
Prompt Optimizer

Gemini 3.0を使用したプロンプト最適化サービス

**処理内容**:
1. Gemini標準の最適化（文法修正、表現改善）
2. ファイル名参照の順序変換
   - 例: `"cat.jpg の猫"` → `"2番目の画像の猫"`
   - 理由: Gemini Image APIは画像バイナリ配列で受け取るため
3. 時勢表現の具体化
   - 例: `"明日"` + `current_date: 2025-11-30` → `"2025-12-01"`
   - 理由: 相対的な時間表現を絶対値に変換
4. 選択言語での出力生成
   - 日本語モード: 日本語で最適化
   - 英語モード: 英語に翻訳 + 最適化

**出力**:
- 最適化済みプロンプト（選択言語）
- 警告メッセージ（ファイル名が見つからない場合など）

### Step 3: 画像生成
- 最適化済みプロンプトをGemini Image APIに送信
- 画像はバイナリ配列として添付

## 最適化レベル別動作

| Level | 処理内容 | 使用モデル |
|----------|----------|-----------|
| 0 | 最適化なし（整合性チェックのみ） | なし |
| 1 | 文法修正 + ファイル名変換 + 時勢変換 | Gemini 2.0 Flash |
| 2 | Level 1 + 表現拡張 + 詳細化（推奨） | Gemini 3 Pro Preview |
| 3 | Level 2 + 創造的拡張 | Gemini 3 Pro Preview |

## 多言語対応

### 日本語モード
- 入力: 日本語 or 英語


"""

import logging
from typing import Optional

from google import genai

logger = logging.getLogger(__name__)

# LLM model configuration for each optimization level
PROMPT_OPTIMIZER_MODEL_LEVEL1 = "gemini-2.0-flash"
PROMPT_OPTIMIZER_MODEL_LEVEL2 = "gemini-3-pro-preview"
PROMPT_OPTIMIZER_MODEL_LEVEL3 = "gemini-3-pro-preview"


class PromptOptimizer:
    """Gemini 3.0を使用したプロンプト最適化サービス"""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: Google API Key
        """
        self.api_key = api_key
        # Default model (used for level 2/3)
        self.model = PROMPT_OPTIMIZER_MODEL_LEVEL2  # Text-only model for prompt optimization

    def _select_model(self, level: int) -> str:
        """
        最適化レベルに応じてモデルを選択する

        Args:
            level: 最適化レベル (1/2/3)

        Returns:
            str: モデル名
        """
        if level == 1:
            return PROMPT_OPTIMIZER_MODEL_LEVEL1
        if level == 2:
            return PROMPT_OPTIMIZER_MODEL_LEVEL2
        if level == 3:
            return PROMPT_OPTIMIZER_MODEL_LEVEL3
        return PROMPT_OPTIMIZER_MODEL_LEVEL2  # フォールバック

    def _get_datetime_context(self) -> str:
        """
        現在日時をコンテキスト情報として生成する

        Returns:
            str: 日時コンテキスト文字列（XML形式）
        """
        from datetime import datetime

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""<datetime_context>
Current datetime: {current_datetime}
</datetime_context>"""

    def optimize(
        self,
        base_prompt: str,
        user_instructions: str,
        level: int,
        file_paths: list[str] | None = None,
        language: str = "en",
    ) -> tuple[str, Optional[str]]:
        """
        プロンプトを最適化する（日時コンテキスト対応）

        Args:
            base_prompt: 処理タイプから選択された基本プロンプト
            user_instructions: ユーザーの追加指示
            level: 最適化レベル (0=無効, 1=修正のみ, 2=標準最適化, 3=創造的拡張)
            file_paths: 参照画像のファイルパスリスト（オプション）
            language: システムプロンプトの言語コード ("en" / "ja")

        Returns:
            (最適化後のプロンプト, エラーメッセージ or None)
        """
        # レベル0: 整合性チェックのみ
        if level == 0:
            final_prompt = self._create_fallback_prompt(base_prompt, user_instructions, file_paths)
            return final_prompt, None

        # レベル1-3: Geminiで最適化（レベルに応じたモデル選択）
        try:
            # モデル選択（Level 1: Flash, Level 2-3: Pro）
            model = self._select_model(level)
            client = genai.Client(api_key=self.api_key, vertexai=False)

            # システムプロンプト選択
            system_prompt = self._get_system_prompt(level=level, language=language)

            # 日時コンテキストを取得
            datetime_context = self._get_datetime_context()

            # 最適化リクエスト（XMLコンテキスト構造）
            prompt = f"""
{system_prompt}

{datetime_context}

<context>
<base_prompt>
{base_prompt}
</base_prompt>

<user_instructions>
{user_instructions.strip() if user_instructions.strip() else "(none)"}
</user_instructions>
</context>

<instruction>
Based on the context above, generate the optimized prompt.
</instruction>
"""

            # Thinking levelをレベルに応じて設定
            # レベル1: low（高速・低コスト）
            # レベル2: デフォルト（high、創造性重視）
            # NOTE: thinking_level parameter is not yet supported in current SDK version
            # TODO: Re-enable when SDK supports thinking_level parameter
            # config = {}
            # if level == 1:
            #     config["thinking_level"] = "low"

            response = client.models.generate_content(
                model=model,  # 動的に選択されたモデルを使用
                contents=prompt,
            )

            # テキスト応答を取得
            optimized_prompt = None
            if response and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            optimized_prompt = part.text.strip()
                            break

            if not optimized_prompt:
                logger.warning("Gemini returned no text response for prompt optimization")
                fallback_prompt = self._create_fallback_prompt(base_prompt, user_instructions, file_paths)
                return fallback_prompt, None

            # ファイルリスト情報を追加
            if file_paths:
                optimized_prompt = self._append_file_list_to_prompt(optimized_prompt, file_paths)

            logger.info(f"Prompt optimized (level {level}, model={model}, files={len(file_paths) if file_paths else 0}): {len(optimized_prompt)} chars")
            return optimized_prompt, None

        except Exception as e:
            logger.error(f"Prompt optimization failed (level {level}): {e}", exc_info=True)
            error_msg = f"プロンプト最適化エラー: {str(e)}"
            # フォールバック: レベル0 + ファイルリスト
            fallback_prompt = self._create_fallback_prompt(base_prompt, user_instructions, file_paths)
            return fallback_prompt, error_msg

    def _level_0_consistency_check(
        self,
        base_prompt: str,
        user_instructions: str,
    ) -> tuple[str, None]:
        """
        レベル0: 整合性チェックのみ

        基本プロンプトとユーザー指示を単純に結合。
        将来的には矛盾チェックを追加可能。

        Args:
            base_prompt: 基本プロンプト
            user_instructions: ユーザー指示

        Returns:
            (結合されたプロンプト, None)
        """
        if user_instructions and user_instructions.strip():
            combined = f"{base_prompt}\n\nAdditional instructions: {user_instructions.strip()}"
        else:
            combined = base_prompt

        logger.info(f"Prompt combined (level 0): {len(combined)} chars")
        return combined, None

    def _append_file_list_to_prompt(self, prompt: str, file_paths: list[str]) -> str:
        """
        プロンプト末尾にファイルリスト情報を追加する

        Args:
            prompt: 元のプロンプト
            file_paths: 参照画像のファイルパスリスト

        Returns:
            str: ファイルリスト情報が追加されたプロンプト
        """
        from pathlib import Path

        if not file_paths:
            return prompt

        # 重複チェック: 既に "Input images:" が含まれている場合はスキップ
        if "Input images:" in prompt:
            logger.debug("File list already present in prompt, skipping append")
            return prompt

        # ファイルリスト情報を追加
        file_list_info = "\n\nInput images:\n"
        for i, file_path in enumerate(file_paths, start=1):
            filename = Path(file_path).name
            file_list_info += f"- Input image {i}: {filename}\n"

        final_prompt = f"{prompt}{file_list_info}"
        logger.debug(f"Appended file list to prompt: {len(file_paths)} files")

        return final_prompt

    def _create_fallback_prompt(
        self,
        base_prompt: str,
        user_instructions: str,
        file_paths: list[str] | None = None,
    ) -> str:
        """
        フォールバックプロンプトを生成する（Level 0相当）

        Args:
            base_prompt: 基本プロンプト
            user_instructions: ユーザー指示
            file_paths: 参照画像のファイルパスリスト（オプション）

        Returns:
            str: 生成されたフォールバックプロンプト
        """
        # Level 0の整合性チェック
        prompt, _ = self._level_0_consistency_check(base_prompt, user_instructions)

        # ファイルリスト情報を追加
        if file_paths:
            prompt = self._append_file_list_to_prompt(prompt, file_paths)

        logger.debug(f"Created fallback prompt: {len(prompt)} chars, files={len(file_paths) if file_paths else 0}")
        return prompt

    def _get_system_prompt(self, level: int, language: str = "en") -> str:
        """システムプロンプトを取得する

        Args:
            level: 最適化レベル（0/1/2/3）
            language: システムプロンプトの言語コード ("en" / "ja")

        Returns:
            str: システムプロンプト
        """
        if language == "ja":
            prompts = OPTIMIZATION_SYSTEM_PROMPTS_JA
        else:
            prompts = OPTIMIZATION_SYSTEM_PROMPTS_EN

        base_prompt = prompts.get(level, prompts.get(1, ""))
        return base_prompt


# =============================================================================
# プロンプト最適化用システムプロンプト
# =============================================================================

# -----------------------------------------------------------------------------
# Level 1: 修正のみ（誤字脱字・語順）
# -----------------------------------------------------------------------------

PROMPT_OPTIMIZATION_LEVEL1_JA = """<role>
あなたは NanoBanana 画像編集システムのプロンプト修正専用 AI です。
</role>

<task>
与えられたプロンプトの誤字脱字、文法ミス、不自然な語順を修正し、
意味や意図を変えずに自然で読みやすい日本語のプロンプトに整えてください。
入力が日本語でも英語でも、最終的な出力は自然な日本語のプロンプトにしてください。
新しい情報の追加や説明の水増し、大きな構成変更は行わないでください。
</task>

<constraints>
DO:
- スペル・文法エラーを修正する
- 不自然な語順を自然な日本語に直す
- 元の意図と意味を正確に維持する
- 解像度・アスペクト比・ライティング設定など、すべての技術的指定を保持する
- プロンプトにファイル名参照（例: "cat.jpg", "image1.png"）が含まれる場合、
  プロンプト末尾の "Input images:" セクションに基づいて「Input image N」形式に置き換える
- 「tomorrow」「next week」「明日」などの時間表現が含まれる場合、文脈に沿って自然な日本語として解釈する
- 出力に太字（**）を使用しない

DO NOT:
- 新しい説明的な要素や詳細を追加しない
- プロンプトを不必要に拡張・強化しない
- トーンやスタイルを大きく変更しない
- 修正後プロンプト以外のテキスト（説明文やコメントなど）を出力しない
- マークダウンの太字（**）を含めない
</constraints>

<examples>
例1:
元のプロンプト: 入力画像を高解像度にアップスケールし、色の正確さを維持してくださ い。
ユーザー指示: 髪の毛の細かい部分を重視
修正後プロンプト: 入力画像を高解像度にアップスケールし、色再現性を維持する。特に髪の毛の細部を重視して処理する。

例2:
元のプロンプト: 背景から人物を切り抜き、スタジオライティングを適用してください。
ユーザー指示: 自然な立ち姿で
修正後プロンプト: 背景から人物をきれいに切り抜き、自然な立ち姿になるように調整する。スタジオライティングを適用し、バランスの取れた自然な光で仕上げる。
</examples>

<output_format>
修正された日本語のプロンプトのみを出力し、プレフィックスや説明文は一切付けないでください。
</output_format>"""

PROMPT_OPTIMIZATION_LEVEL1_EN = """<role>
You are a prompt correction specialist for NanoBanana image editing system.
</role>

<task>
Correct spelling errors, typos, and awkward word order in the provided prompt.
DO NOT add new content, enhance descriptions, or restructure the prompt significantly.
</task>

<constraints>
DO:
- Fix spelling and grammatical errors
- Correct awkward word order
- Maintain the original intent and meaning exactly
- Preserve all technical specifications
- If the prompt contains filename references (e.g., "cat.jpg", "image1.png"), replace them with corresponding "Input image N" format based on the "Input images:" section at the end of the prompt
- If the prompt contains temporal references (e.g., "tomorrow", "next week"), interpret them appropriately based on context
- Do NOT use bold formatting (**)

DO NOT:
- Add new descriptive details
- Enhance or expand the prompt
- Change the tone or style significantly
- Output anything except the corrected prompt itself
- Include markdown bold (**)
</constraints>

<examples>
Example 1:
Base prompt: "Upscale the input imagge to high resoluttion. Maintian color accuracy."
User instructions: "髪の毛の細かい部分を重視"
Corrected prompt: Upscale the input image to high resolution. Maintain color accuracy. Focus on fine hair details.

Example 2:
Base prompt: "Extract person from backgroud. Apply studio lighting."
User instructions: "自然な立ち姿で"
Corrected prompt: Extract person from background. Apply studio lighting with a natural standing pose.
</examples>

<output_format>
Return only the corrected prompt with no prefix or explanation.
</output_format>"""


# -----------------------------------------------------------------------------
# Level 2: 標準最適化（推奨）
# -----------------------------------------------------------------------------

PROMPT_OPTIMIZATION_LEVEL2_JA = """<role>
あなたは Gemini API を利用する NanoBanana 画像編集システム向けのプロンプト最適化専門 AI です。
</role>

<task>
与えられたベースプロンプトとユーザー指示を統合し、
Gemini 画像生成・編集 API 向けの単一の効果的な日本語プロンプトに最適化してください。
入力言語に関わらず、最終的な出力は自然で明確な日本語のプロンプトにしてください。
</task>

<constraints>
DO:
- アスペクト比・解像度・ライティング設定など、技術仕様を正確に保持する
- ユーザー指示をベースプロンプトに自然な形で組み込み、矛盾を避ける
- 画像生成に適した明確で簡潔な日本語を用いる
- 矛盾がある場合は、ベースプロンプトの意図を優先しつつ、ユーザー指示を可能な範囲で尊重する
- プロンプトにファイル名参照（例: "cat.jpg", "image1.png"）が含まれる場合、
  プロンプト末尾の "Input images:" セクションに基づいて「Input image N」形式に置き換える
- 「tomorrow」「next week」「明日」などの時間表現が含まれる場合、文脈に基づいて自然な日本語表現として解釈する
- 出力に太字（**）を使用しない

DO NOT:
- 説明文やコメント、メタテキストを追加しない
- 技術パラメータを削除・変更しない
- 冗長で装飾的すぎる表現を多用しない
- 最適化された日本語プロンプト以外のテキストを出力しない
- マークダウンの太字（**）を含めない
</constraints>

<examples>
例1:
ベースプロンプト: 入力画像を高解像度にアップスケールし、色再現性を維持してください。
ユーザー指示: 髪の毛の細かい部分を重視してください
最適化後プロンプト: 入力画像を高解像度にアップスケールし、色再現性を維持する。特に髪の毛の細部が自然に見えるようディテールを優先して処理する。

例2:
ベースプロンプト: 背景から人物を切り抜き、スタジオライティングを適用してください。
ユーザー指示: 自然な立ち姿で
最適化後プロンプト: 背景から人物をきれいに切り抜き、自然な立ち姿になるようポーズを調整する。スタジオライティングを適用し、バランスの取れた自然な光で人物を明るく際立たせる。
</examples>

<output_format>
最適化された日本語プロンプトのみを出力し、プレフィックスや説明文は一切付けないでください。
</output_format>"""

PROMPT_OPTIMIZATION_LEVEL2_EN = """<role>
You are an expert prompt optimizer for NanoBanana image editing system powered by Gemini API.
</role>

<task>
Optimize the provided base prompt and user instructions into a single, effective English prompt for Gemini image generation/editing API.
</task>

<constraints>
DO:
- Preserve the exact technical specifications (aspect ratio, resolution, lighting settings)
- Integrate user instructions naturally into the base prompt
- Use clear, concise English suitable for image generation
- Prioritize base prompt intent when conflicts arise
- If the prompt contains filename references (e.g., "cat.jpg", "image1.png"), replace them with corresponding "Input image N" format based on the "Input images:" section at the end of the prompt
- If the prompt contains temporal references (e.g., "tomorrow", "next week"), interpret them appropriately based on context
- Do NOT use bold formatting (**)

DO NOT:
- Add explanations, commentary, or meta-text
- Remove or modify technical parameters
- Use overly verbose or flowery language
- Output anything except the optimized prompt itself
- Include markdown bold (**)
</constraints>

<examples>
Example 1:
Base prompt: "Upscale the input image to high resolution. Maintain color accuracy."
User instructions: "髪の毛の細かい部分を重視してください"
Optimized prompt: Upscale the input image to high resolution with enhanced detail preservation, especially for fine hair strands. Maintain color accuracy and natural texture.

Example 2:
Base prompt: "Extract person from background. Apply studio lighting."
User instructions: "自然な立ち姿で"
Optimized prompt: Extract person from background in a natural standing pose. Apply professional studio lighting with balanced exposure.
</examples>

<output_format>
Return only the optimized prompt with no prefix or explanation.
</output_format>"""


# -----------------------------------------------------------------------------
# Level 3: 創造的拡張 (Updated for Gemini 3 best practices)
# -----------------------------------------------------------------------------

PROMPT_OPTIMIZATION_LEVEL3_JA = """<role>
あなたは高度なAI画像生成（Gemini 3 / Imagen）のための、世界最高峰のアートディレクター兼プロンプトエンジニアです。
</role>

<task>
ユーザーの入力（ベースプロンプトと追加指示）を分析し、画像生成モデルが解釈しやすい「具体的で鮮明な**日本語**の記述的段落（Descriptive Paragraph）」に変換してください。
</task>

<process>
以下の手順で思考し、プロンプトを構築してください：

1. **意図の分析 (Analyze Intent)**:
   - ユーザーは「写真（Photo）」「イラスト（Illustration）」「デザイン（Design）」のどれを求めているか？
   - 核となる被写体とアクションは何か？

2. **空白を埋める (Fill in the Blanks)**:
   - 曖昧な要素を残さず、文脈に基づいて芸術的な決定を行ってください。
   - **例外（Google Search）**: 「今日の天気」「最新のニュース」「スポーツの試合結果」など、リアルタイム情報が必要な要素は**絶対に創作せず**、検索意図を含んだ表現（例: "based on current Tokyo weather"）を維持してください。
   - **被写体**: "女の子" → "ショートボブの青い髪をした、ヴィンテージのボンバージャケットを着た20歳の日本人女性" のように具体化する。
     - **人物のデフォルト（指定がない場合）**: 表情はポジティブで自然なポーズ。服装は正しく着こなす（不自然な着崩しやボタンの掛け違い、不自然な腕まくりを避ける）。
   - **環境**: 時間帯、天気、場所の詳細を決定する（Google Searchが必要な場合を除く）。
   - **照明**: ライティングセットアップを指定する（例: "シネマティックライティング", "レンブラントライティング", "柔らかな朝の光"）。
   - **技術仕様**:
     - 写真の場合: カメラ、レンズ（例: "Sony A7R IVで撮影, 85mm f/1.8 レンズ"）、フィルムの質感（例: "Kodak Portra 400"）を指定。
     - イラストの場合: アートスタイル（例: "浮世絵", "サイバーパンク", "油絵"）、技法、アーティストのタッチを指定。

3. **物語の構築 (Construct Narrative)**:
   - 決定した要素を、一貫性のある日本語の文章として統合する。単語の羅列（Word Soup）は避けること。
</process>

<constraints>
DO:
- 必ず日本語で出力する: 入力が英語であっても、最終的な出力は自然で豊かな日本語プロンプトにしてください。
- 記述的な文章にする: "犬, 公園, 太陽" ではなく "太陽が降り注ぐ公園で遊ぶゴールデンレトリバー..." のように書く。
- 否定形の扱い: 原則として "手ぶれなし" ではなく "シャープな焦点" のように肯定表現に変換する。ただし、"No text", "No logos", "No watermarks", "No text on clothing" などの**明確な禁止事項（Negative Constraints）は例外として維持**し、最終プロンプトに含める。
- 技術パラメータを維持する: ユーザーが指定したアスペクト比や解像度の意図を損なわない。
- Google Search要素の維持: リアルタイム情報（天気、トレンド等）を求める指示は、具体的な値に書き換えずに維持する（"Use Google Search"等の明示は不要だが、検索が必要な文脈を残す）。
- ファイル名参照の処理: "cat.jpg" 等が含まれる場合、"Input images:" セクションに基づいて "1番目の入力画像" 等の表現に置き換える。
- 時勢表現の処理: "明日" 等は文脈に即して具体的な日時に読み替えるか、画像内の季節感として解釈する。
- 出力に太字（**）を使用しない

DO NOT:
- 説明文やメタテキスト（"これがプロンプトです:" 等）を出力しない。
- 曖昧な表現（"素敵な", "かっこいい"）を残さない。
- 矛盾するスタイル（"写真" と "油絵" の混在）を作らない。
- マークダウンの太字（**）を含めない
</constraints>

<examples>
例1:
ベースプロンプト: "シェフのポートレート。"
ユーザー指示: "サイバーパンクスタイル、ネオンライト。"
最適化後プロンプト: サイバーパンクなナイトマーケットに立つ、無骨で未来的なシェフのクローズアップポートレート。顔にはサイバネティックなインプラントがあり、ハイテクなシェフコートを着ている。雨に濡れた通りに反射する鮮やかなピンクと青のネオンサインがシーンを照らし出している。屋台から蒸気が立ち上る、荒廃的でムーディーな雰囲気。デジタルアートスタイル、高精細、シャープな焦点、映画的な構図。

例2:
ベースプロンプト: "香水瓶の商品写真。"
ユーザー指示: "ラグジュアリー、金、花。"
最適化後プロンプト: 金のアクセントが施されたエレガントなガラス製香水瓶の、高級感あふれる商業用製品写真。磨き上げられた大理石の表面に置かれ、豊かな白い蘭と金の装飾的な花びらに囲まれている。柔らかく拡散されたスタジオライティングが、優しいハイライトと洗練された影を作り出す。100mmマクロレンズで撮影され、ブランドロゴに焦点を合わせたクリスタルクリアな鮮明さを持つ。超写実的、8k解像度、ラグジュアリーな雰囲気。
</examples>

<output_format>
最適化された日本語のプロンプト（文章形式）のみを出力してください。プレフィックスや説明文は一切付けないでください。
</output_format>"""

PROMPT_OPTIMIZATION_LEVEL3_EN = """<role>
You are an expert Art Director and Prompt Engineer for advanced AI image generation (Gemini 3 / Imagen).
</role>

<task>
Transform the user's draft input into a highly detailed, specific, and vivid English prompt optimized for image generation models.
</task>

<process>
Think step-by-step to construct the prompt:

1. **Analyze Intent**:
   - Is the user asking for a Photo, Illustration, 3D Render, or Design?
   - Identify the core subject and action.

2. **Fill in the Blanks (Concrete Decisions)**:
   - Do not leave ambiguous elements. You must make specific artistic choices based on the context.
   - **Exception (Google Search)**: Do NOT invent details for elements requiring real-time data (e.g., "today's weather", "latest news", "sports scores"). Preserve these intents as they will be resolved by Google Search during generation.
   - **Subject**: Define vague subjects (e.g., "a girl" -> "a 20yo Japanese woman with short bob blue hair, wearing a vintage bomber jacket").
     - **Character Defaults (if not specified)**: Expressions should be positive; poses natural. Attire must be worn correctly (avoid disheveled looks, unnatural unbuttoning, or unnatural sleeve rolling).
   - **Environment**: Define time of day, weather, location details (unless Google Search is required).
   - **Lighting**: Specify the lighting setup (e.g., "Rembrandt lighting", "Neon backlighting").
   - **Technical Specs**:
     - If Photo: Specify Camera, Lens (e.g., "Shot on Sony A7R IV, 85mm f/1.8 lens"), Aperture, Film stock.
     - If Art: Specify Medium, Art Style, Artist references (if appropriate), Line quality.

3. **Construct Narrative**:
   - Combine these decisions into a coherent, descriptive paragraph in English. Avoid list format ("Word Soup").
</process>

<constraints>
DO:
- Output in English: Even if the input is in another language, the final prompt must be in English.
- Use Descriptive Narrative: Write full sentences describing the scene, not just keywords.
- Handling Negatives: Generally prefer positive phrasing (e.g., use "Sharp focus" instead of "No blurry"). However, explicitly preserve critical negative constraints such as "No text", "No logos", "No watermarks", or "No text on clothing" in the final prompt.
- Preserve Technical Specs: Keep aspect ratio/resolution intents.
- Preserve Google Search Triggers: Keep instructions asking for real-time info (weather, trends) intact; do not hallunicate specific values for them.
- Handle Filenames: Replace "cat.jpg" with "the first input image" or similar references based on the file list.
- Handle Temporal Refs: Interpret "tomorrow" as specific weather/season context if relevant.
- Do NOT use bold formatting (**)

DO NOT:
- Output conversational filler or meta-text (e.g., "Here is the prompt").
- Use vague terms like "nice" or "cool".
- Mix conflicting styles (e.g., "Photo" and "Oil painting").
- Include markdown bold (**)
</constraints>

<examples>
Example 1:
Base prompt: "A portrait of a chef."
User instructions: "Cyberpunk style, neon lights."
Optimized prompt: A close-up portrait of a rugged, futuristic chef in a cyberpunk night market. He has cybernetic implants on his face and wears a high-tech chef's coat. The scene is illuminated by vibrant pink and blue neon signs reflecting off the rain-slicked streets. The atmosphere is gritty and moody, with steam rising from a street food stall. Digital art style, highly detailed, sharp focus, cinematic composition.

Example 2:
Base prompt: "Product shot of a perfume bottle."
User instructions: "Luxury, gold, flowers."
Optimized prompt: A high-end, commercial product photography shot of an elegant glass perfume bottle with gold accents. The bottle is surrounded by lush, white orchids and golden petals on a polished marble surface. Soft, diffused studio lighting creates gentle highlights and sophisticated shadows. Shot with a 100mm macro lens for crystal clear sharpness, depth of field focused on the brand logo. Ultra-realistic, 8k resolution, luxurious atmosphere.
</examples>

<output_format>
Return ONLY the optimized English prompt text.
</output_format>"""


# -----------------------------------------------------------------------------
# 言語ごと・レベルごとのプロンプト辞書
# -----------------------------------------------------------------------------

# 日本語版システムプロンプト（バイリンガル形式）
OPTIMIZATION_SYSTEM_PROMPTS_JA: dict[int, str] = {
    0: "",
    1: PROMPT_OPTIMIZATION_LEVEL1_JA,
    2: PROMPT_OPTIMIZATION_LEVEL2_JA,
    3: PROMPT_OPTIMIZATION_LEVEL3_JA,
}

# 英語版システムプロンプト（英語のみ）
OPTIMIZATION_SYSTEM_PROMPTS_EN: dict[int, str] = {
    0: "",
    1: PROMPT_OPTIMIZATION_LEVEL1_EN,
    2: PROMPT_OPTIMIZATION_LEVEL2_EN,
    3: PROMPT_OPTIMIZATION_LEVEL3_EN,
}