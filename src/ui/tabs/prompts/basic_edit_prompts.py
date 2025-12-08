"""
Basic Edit Tab Prompts

ベーシック編集タブで使用する全プロンプト定義
このファイルはbasic_edit_tabでのみ使用される（他タブとの共有は影響度が高いため行わない）
"""

# =============================================================================
# 編集プロンプト定義
# =============================================================================

# 生成アップスケールのみ - シンプルな1回編集専用
# 参考訳：
# 入力画像を指定された解像度とアスペクト比に合わせて高品質にアップスケールしなさい。
# 画像の内容、構図、色調は元のまま保持すること。
# ディテールを自然に補完し、シャープで鮮明な高解像度画像を出力しなさい。
# アーティファクトやノイズを最小限に抑え、プロフェッショナルな品質を維持すること。
UPSCALE_ONLY_PROMPT = (
    "Upscale the input image to the specified resolution and aspect ratio with high quality.  "
    "Preserve the original content, composition, and color tone of the image.  "
    "Naturally supplement details and output a sharp, clear high-resolution image.  "
    "Minimize artifacts and noise while maintaining professional quality."
)

# 色温度調整 - 新規追加
# 参考訳：
# 入力画像の色温度を最適化し、自然でバランスの取れた色調に調整しなさい。
# 被写体の肌のトーン、環境光、全体的な雰囲気を考慮すること。
# 暖色（オレンジ系）または寒色（ブルー系）のバイアスを適切に補正し、
# プロフェッショナルな品質を維持しながら自然な仕上がりにすること。
# 画像のシャープネスとディテールはそのまま保持すること。
COLOR_TEMPERATURE_PROMPT = (
    "Optimize the color temperature of the input image to achieve a natural and balanced color tone.  "
    "Consider the subject's skin tone, ambient lighting, and overall atmosphere.  "
    "Appropriately correct any warm (orange) or cool (blue) bias, "
    "while maintaining professional quality and a natural finish.  "
    "Preserve the image's sharpness and detail."
)
