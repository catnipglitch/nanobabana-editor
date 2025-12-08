"""
プロンプト最適化デモ用プロンプト集

プロンプト最適化関数が正しく動作しているか確認するためのデモ用プロンプトを定義します。
実行すると用意したプロンプトに対し、レベル１，２，３で最適化を行い、その結果を表示します。
結果はoutputディレクトリに.mdファイルとして保存されます。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

from .prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


# name = "アニメ-高校生女子-設定資料"
demo_prompt_001_e = """
Generate a full-body character reference sheet (settei) for a Japanese high school girl.
Subject: Female student in a standard school uniform. Random hairstyle and hair color.
Pose: Clean, standing front view (turnaround-friendly) suitable for 3D modeling or animation reference.
Style: Standard Japanese anime style, clear lineart, simple cel-shading.
Background: Plain white.
Constraints: High-quality asset for production. No text, no logos, no watermarks. No text on clothing or accessories.
"""

demo_prompt_001_j = """

日本の女子高校生の全身キャラクター設定資料（設定画）を生成してください。
被写体: 標準的な制服を着た女子生徒。髪型と髪色はランダム。
ポーズ: 3Dモデリングやアニメーションの参考資料に適した、清潔感のある正面立ち絵（ターンアラウンド対応）。
スタイル: 標準的な日本のアニメスタイル、明確な線画、シンプルなセルシェーディング。
背景: 無地の白。
制約: 制作向けの高品質素材。テキスト、ロゴ、透かしは禁止。衣装やアクセサリーへの文字入れも禁止。
"""
demo_prompt_002_j = """

# 前提条件
地域・場所: 東京 渋谷スクランブル交差点

# 指示
上記「前提条件」で指定された場所にて、テレビの天気予報番組が生中継を行っている画像を生成してください。
Google検索を使用し、実在の最新情報を反映させること。

## STEP 1: Google検索 (データ取得)
指定された場所について以下の検索を行い、情報を取得してください。
1. 向こう3日間の天気予報 (日付、曜日、天気、最高/最低気温)
2. 現在の天気と適した服装
3. 現在時刻 (現地時間)

## STEP 2: 変数定義
検索結果に基づき、以下の要素を確定させてください。
- {CURRENT_TIME}: 画像生成時点の時刻 (例: 14:35)。
- {CURRENT_WEATHER}: 現在の現地の天候、空の明るさ、ライティング。
- {OUTFIT_STYLE}: 気温と天気に適した、清潔感のあるキャスター風のモダンな服装。
- {FORECAST_DATA}: 明日から3日間の天気情報。

## STEP 3: 画像生成
以下の要件で画像を描画してください。

### 1. テーマと構図
- テーマ: 天気予報番組の生中継
- 構図: 現地からレポートするお天気キャスター。バストアップからウェストアップのアングル。臨場感のあるバランス。
- UIオーバーレイ:
    - 左上: 時刻 {CURRENT_TIME} と番組ロゴ "NHTV" (白文字、ドロップシャドウ、サンセリフ体)
    - 右上: 赤い "LIVE" アイコン と "東京・渋谷" (白文字、日本語表記)

### 2. メインキャラクター
- キャラクター: アニメ調のかわいいお天気お姉さん（Live2D/Vtuberスタイル）
- 演技: カメラ（視聴者）に向かってプロらしく、天気予報説明パネルを丁寧に両手で自然に持ち、親しみやすく語りかけている。
- 服装: {OUTFIT_STYLE} (季節感とトレンドを意識)

### 3. アイテム：天気予報パネル
- アイテム: 情報が整理された天気予報パネル。必ず日本語を使用すること。
- 内容: 横並びの3つの枠。日付(曜日: 月,火など) | 天気アイコン | 最高/最低気温 の形式。
- フォント: 太字で視認性を高く。日本語フォント。
- 注記: "明日"などの文字は避け、具体的な日付を使う。
    検索結果に基づき、日本語の手書き風ワンポイントアドバイスを「1つだけ」下部に追加すること。
    例:
    - 雨: "傘が必要です"
    - 晴れ: "洗濯日和です"
    - 暑い: "熱中症に注意"
    - 寒い: "暖かくしてね"
    - 強風: "強風に注意"

### 4. 背景
- 場所: 渋谷スクランブル交差点の風景。
- 群衆: 通行人は存在するが、遠景や中景に配置し、キャスターの周囲は空ける。
- 被写界深度: 背景の街並みと通行人は深くぼかす（ボケ味）。キャスターのみにピント。
- ライティング: {CURRENT_WEATHER} と {CURRENT_TIME} を反映したリアルな環境光。キャラクターにも適用。
制約: 高品質な素材。服への文字入れ禁止。

"""

# 複数画像参照テスト用プロンプト
demo_prompt_003_multi_image_j = """
Input image 2 のポーズを参考にして、Input image 3 のアートスタイルで、Input image 1 のキャラクターを Input image 4 のデスクに座らせて描画してください。

要件:
- キャラクター:  (char.jpg) のキャラクターの外見を保持
- ポーズ:  (pose_ref.jpg) のポーズを再現
- アートスタイル:  (style_ref.jpg) のアートスタイル（線画、色調、シェーディング）を適用
- 背景:  (desk.png) のデスク環境を背景として使用
- 構図: キャラクターがデスクに自然に座っている様子
- ライティング: デスク環境に合わせた自然な照明

Input images:
- Input image 1: char.jpg
- Input image 2: pose_ref.jpg
- Input image 3: style_ref.jpg
- Input image 4: desk.png
"""
# 複数画像参照テスト用プロンプト
demo_prompt_003_multi_image_e = """
Draw the character from Input image 1 sitting at the desk from Input image 4, using the pose from Input image 2 and the art style from Input image 3.

Requirements:
- Character: Preserve the appearance of the character from (char.jpg)
- Pose: Reproduce the pose from Input image 
- Art Style: Apply the art style (linework, color palette, shading) from (style_ref.jpg)
- Background: Use the desk environment from Input  (desk.png) as the background
- Composition: The character should be naturally seated at the desk
- Lighting: Use natural lighting that matches the desk environment

Input images:
- Input image 1: char.jpg
- Input image 2: pose_ref.jpg
- Input image 3: style_ref.jpg
- Input image 4: desk.png
"""


# デモプロンプトのリスト
DEMO_PROMPTS = {
    "001_english": {
        "name": "アニメ-高校生女子-設定資料（英語）",
        "prompt": demo_prompt_001_e,
    },
    "001_japanese": {
        "name": "アニメ-高校生女子-設定資料（日本語）",
        "prompt": demo_prompt_001_j,
    },
    "002_japanese": {
        "name": "天気予報番組生中継（日本語・複雑）",
        "prompt": demo_prompt_002_j,
    },
    "003_multi_image_j": {
        "name": "複数画像参照テスト（キャラ×ポーズ×スタイル×背景）",
        "prompt": demo_prompt_003_multi_image_j,
    },
    "003_multi_image_e": {
        "name": "複数画像参照テスト（キャラ×ポーズ×スタイル×背景）",
        "prompt": demo_prompt_003_multi_image_e,
    },
}

# デモテスト定義リスト（プロンプト×レベル×言語の組み合わせ）
# コメントアウトして個別に無効化可能
# 形式: (prompt_key, level, language)
# language: "ja" (日本語出力) or "en" (英語出力)
DEMO_TESTS = [
    # デモ001（英語）- 全レベル
    #    ("001_english", 1, "en"),
    #    ("001_english", 2, "en"),
    #    ("001_english", 3, "en"),
    # デモ001（日本語）- 全レベル
    #    ("001_japanese", 1, "ja"),
    #    ("001_japanese", 2, "ja"),
    #    ("001_japanese", 3, "ja"),
    # デモ002（日本語・複雑）- レベル2と3のみ
    ("002_japanese", 2, "ja"),
    ("002_japanese", 3, "ja"),
    # デモ003（複数画像参照テスト）- レベル2と3で試す
    ("003_multi_image_e", 2, "en"),
    ("003_multi_image_j", 2, "ja"),
    #    ("003_multi_image_e", 3, "en"),
    #    ("003_multi_image_j", 3, "ja"),
    # 必要に応じて追加・削除・コメントアウト可能
]


def run_all_optimization_demos(
    api_key: str,
    output_dir: Path | None = None,
) -> Tuple[str, str]:
    """
    全デモテストを一括実行してプロンプト最適化の動作を確認

    Args:
        api_key: Google API Key
        output_dir: 出力ディレクトリ（デフォルト: output/）

    Returns:
        (結果テキスト, 保存ファイルパス): Markdown形式の結果サマリーと保存先パス
    """
    logger.info("=== Running All Optimization Demos ===")

    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ付きファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"prompt_opt_demo_{timestamp}.md"
    output_path = output_dir / output_filename

    # レベル名のマッピング
    level_names = {
        1: "レベル1: 修正のみ",
        2: "レベル2: 標準最適化",
        3: "レベル3: 創造的拡張",
    }

    # PromptOptimizer初期化
    optimizer = PromptOptimizer(api_key)

    # 全テスト結果を蓄積
    test_results = []
    total_tests = len(DEMO_TESTS)
    successful_tests = 0
    failed_tests = 0

    logger.info(f"Total tests to run: {total_tests}")

    # 各テストを実行
    for idx, (prompt_key, level, language) in enumerate(DEMO_TESTS, 1):
        # プロンプト情報を取得
        if prompt_key not in DEMO_PROMPTS:
            logger.warning(
                f"Test {idx}/{total_tests}: Invalid prompt_key '{prompt_key}', skipping"
            )
            failed_tests += 1
            test_results.append(
                {
                    "index": idx,
                    "prompt_key": prompt_key,
                    "level": level,
                    "error": f"無効なプロンプトキー: {prompt_key}",
                }
            )
            continue

        demo_info = DEMO_PROMPTS[prompt_key]
        prompt_name = demo_info["name"]
        original_prompt = demo_info["prompt"]
        level_name = level_names.get(level, f"レベル{level}")

        logger.info(
            f"Test {idx}/{total_tests}: {prompt_name} - {level_name} (language: {language})"
        )

        try:
            # プロンプト最適化を実行
            optimized_prompt, error = optimizer.optimize(
                base_prompt="",
                user_instructions=original_prompt,
                level=level,
                language=language,
            )

            if error:
                logger.warning(
                    f"Test {idx}/{total_tests}: Optimization warning: {error}"
                )

            test_results.append(
                {
                    "index": idx,
                    "prompt_name": prompt_name,
                    "level": level,
                    "level_name": level_name,
                    "original_prompt": original_prompt,
                    "optimized_prompt": optimized_prompt,
                    "error": error,
                }
            )
            successful_tests += 1

        except Exception as e:
            logger.error(
                f"Test {idx}/{total_tests}: Optimization failed: {e}", exc_info=True
            )
            failed_tests += 1
            test_results.append(
                {
                    "index": idx,
                    "prompt_name": prompt_name,
                    "level": level,
                    "level_name": level_name,
                    "original_prompt": original_prompt,
                    "error": f"最適化エラー: {str(e)}",
                }
            )

    # Markdown結果の構築
    md_lines = []
    md_lines.append("# プロンプト最適化デモ結果（全テスト一括実行）")
    md_lines.append("")
    md_lines.append(f"- **実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- **実行テスト数**: {total_tests}件")
    md_lines.append(f"- **成功**: {successful_tests}件 / **失敗**: {failed_tests}件")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 各テスト結果を出力
    for result in test_results:
        idx = result["index"]
        md_lines.append(
            f"## テスト {idx}/{total_tests}: {result.get('prompt_name', 'Unknown')} - {result.get('level_name', 'Unknown')}"
        )
        md_lines.append("")

        # エラーがある場合
        if "error" in result and result["error"] and "optimized_prompt" not in result:
            md_lines.append(f"**❌ エラー**: {result['error']}")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
            continue

        # 元のプロンプト
        md_lines.append("### 元のプロンプト")
        md_lines.append("")
        md_lines.append("```")
        md_lines.append(result["original_prompt"].strip())
        md_lines.append("```")
        md_lines.append("")
        md_lines.append(f"文字数: {len(result['original_prompt'])} chars")
        md_lines.append("")

        # 最適化後プロンプト
        md_lines.append(f"### 最適化後プロンプト（{result['level_name']}）")
        md_lines.append("")
        md_lines.append("```")
        md_lines.append(result["optimized_prompt"].strip())
        md_lines.append("```")
        md_lines.append("")
        md_lines.append(f"文字数: {len(result['optimized_prompt'])} chars")
        md_lines.append("")

        # 警告がある場合
        if result.get("error"):
            md_lines.append(f"**⚠️ 警告**: {result['error']}")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    # ファイルに保存
    md_content = "\n".join(md_lines)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"All demo results saved to: {output_path}")
        save_info = f"✅ 結果を保存しました: `{output_path.name}`"
    except Exception as e:
        logger.error(f"Failed to save demo results: {e}")
        save_info = f"⚠️ 保存失敗: {str(e)}"

    # 結果サマリーの生成
    result_summary = f"""### 全デモ実行完了

{save_info}

**実行結果**:
- 成功: {successful_tests}件
- 失敗: {failed_tests}件
- 合計: {total_tests}件

詳細は保存されたファイルをご確認ください。
"""

    return result_summary, str(output_path)
