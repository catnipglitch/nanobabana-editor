"""
Basic Edit Tab Prompts

ベーシック編集タブで使用する全プロンプト定義
このファイルはbasic_edit_tabでのみ使用される（他タブとの共有は影響度が高いため行わない）
"""

# =============================================================================
# 編集プロンプト定義
# =============================================================================

# パターン１: 元の写真を忠実に活かす・切り抜き＋元のライティング＋姿勢そのまま
# 入力画像1は人物のポートレートです。
# 広告素材として利用できるよう、人物を丁寧に切り抜き、背景の要素をすべて取り除きなさい。
# ライティングは元の写真の状態を忠実に再現し、光の方向、強さ、色温度、肌の反射などを変更してはならない。
# 人物の姿勢や構図は元のまま保持し、体型・表情・衣服の色を変えずに自然に保ちなさい。
# 人物の輪郭や髪の毛の細部は滑らかに処理し、透明背景でノイズのない広告向けの素材として仕上げなさい。
# 最終的な画像は指定された出力サイズに合わせて高品質にアップスケールしなさい。

UPSCALE_PROMPT_1 = (
    "Input Image 1 is a portrait of a person.  "
    "Prepare it as an advertising asset by carefully extracting the person and removing all background elements.  "
    "Reproduce the original lighting faithfully, without altering the direction, intensity, color temperature, or reflections on the skin.  "
    "Keep the posture and composition exactly as in the original image, preserving the person's body shape, expression, and clothing colors.  "
    "Maintain smooth edges and detailed hair strands, and output a clean, transparent background suitable for compositing.  "
    "Finally, upscale the image to match the required output resolution with high quality."
)


# パターン２: 切り抜き＋スタジオ風ライティング化（姿勢はそのまま）
# 入力画像1は人物のポートレートです。
# 広告素材として利用できるよう、人物のみを丁寧に切り抜き、背景を完全に取り除きなさい。
# ライティングは均一でフラットなスタジオ光に整え、影の方向性が強まる照明やレンブラントライトのような演出を避けること。
# 人物はスタジオで自然に立っている姿勢として再構築しなさい。
# 両足に重心を均等に置き、腕は身体から少し離し、表情はナチュラルでニュートラルな素材向けとすること。
# 元の写真に写っていない下半身や手足などがある場合は、人物の体格と雰囲気に合わせて自然に想像し補完しなさい。
# 人物の特徴や衣服の色は変えず、広告合成に適した透明背景で仕上げること。
# 最終画像は出力サイズに合わせて高品質にアップスケールしなさい。

UPSCALE_PROMPT_2 = (
    "Input Image 1 is a portrait of a person.  "
    "Extract the person cleanly and remove the background to prepare the subject for advertising use.  "
    "Adjust the lighting to a flat, even studio style, avoiding dramatic shadows or directional highlights.  "
    "Ensure the illumination and color temperature remain neutral so the figure blends naturally with any presentation or design material.  "
    "Preserve the original posture and composition without altering the subject's features or color tones.  "
    "Finally, upscale the image to the required output size with high resolution."
)


# パターン３: 切り抜き＋スタジオ立ち姿（ポーズ補完あり）
# ※見えない部分は自然に再構築する広告特化モデル
# 入力画像1は人物のポートレートです。
# 広告素材として利用できるよう、人物のみを丁寧に切り抜き、背景を完全に取り除きなさい。
# ライティングは均一でフラットなスタジオ光に整え、影の方向性が強まる照明やレンブラントライトのような演出を避けること。
# 人物はスタジオで自然に立っている姿勢として再構築しなさい。
# 両足に重心を均等に置き、腕は身体から少し離し、表情はナチュラルでニュートラルな素材向けとすること。
# 元の写真に写っていない下半身や手足などがある場合は、人物の体格と雰囲気に合わせて自然に想像し補完しなさい。
# 人物の特徴や衣服の色は変えず、広告合成に適した透明背景で仕上げること。
# 最終画像は出力サイズに合わせて高品質にアップスケールしなさい。

UPSCALE_PROMPT_3 = (
    "Input Image 1 is a portrait of a person.  "
    "Extract the subject cleanly and remove all background elements to prepare the figure for advertising use.  "
    "Adjust the lighting to a flat, uniform studio style, avoiding dramatic shadows or Rembrandt-style highlights.  "
    "Reconstruct the person as standing naturally in a studio setting, with weight evenly distributed on both feet, arms slightly away from the body, and a neutral, natural expression suitable for commercial assets.  "
    "If any body parts are missing—such as the lower body, hands, or legs—extend and complete them naturally based on the subject's physique and appearance.  "
    "Preserve the subject's inherent features and clothing colors, and output a clean transparent background for easy compositing.  "
    "Finally, upscale the image to the required output resolution with high quality."
)

# シンプルなアップスケールプロンプト（未使用、サンプルコードから移植）
UPSCALE_PROMPT_4 = ""


# =============================================================================
# アルファマットプロンプト定義
# =============================================================================

# キャラクター用（イラスト）
# 参考訳：
# 入力画像のキャラクターの、プロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# キャラクターの体、髪、装着しているアクセサリーは白でレンダリングしてください。
# 重要：半透明部分（細い髪の毛、レース生地、透明プラスチックなど）は、
# 不透明度レベルを示す正確なグレーの階調を使用して表現してください。
# 影、床の反射、床に置かれた切り離されたオブジェクト（かばんや被っていない帽子など）は
# 厳密に除外してマスクアウトしてください。
# 出力は、プロフェッショナルな合成に適した、クリーンでノイズのないセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_CHARACTER = (
    "A professional high-precision grayscale alpha matte of the character from the input image. "
    "The background must be pure black (Hex #000000). "
    "The character's body, hair, and worn accessories must be rendered in white. "
    "Crucially, represent semi-transparency (e.g., fine hair strands, lace fabric, transparent plastic) "
    "using accurate shades of gray to indicate opacity levels. "
    "Strictly exclude and mask out any cast shadows, floor reflections, and detached objects placed on the floor "
    "(such as bags or hats not being worn). "
    "The output should be a clean, noise-free segmentation mask suitable for professional compositing."
)

# 人物用（実写）- V1
# 参考訳：
# 入力画像の人物の、プロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# 人物の肌（顔、腕、脚、体）は、ソリッドな不透明の白でレンダリングしてください - 肌は決して透明ではありません。
# 髪、衣服、アクセサリーは白でレンダリングしてください。
# グレーの階調を使用した半透明表現は、次の場合のみ適用してください：
# - エッジの細い髪の毛
# - 透明またはシアーな生地（レース、チュール、オーガンザ）
# - 透明なアクセサリー（眼鏡、プラスチック製品）
# 肌、ソリッドな衣服、メインボディは常にソリッドな不透明の白である必要があります。
# 影、床の反射、床に置かれた切り離されたオブジェクトは厳密に除外してマスクアウトしてください。
# 出力は、プロフェッショナルな合成に適した、クリーンでノイズのないセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_HUMAN = (
    "A professional high-precision grayscale alpha matte of the person from the input image. "
    "The background must be pure black (Hex #000000). "
    "The person's skin (face, arms, legs, body) must be rendered as solid opaque white - skin is never transparent. "
    "Hair, clothing, and accessories must be rendered in white. "
    "Semi-transparency using gray shades should ONLY be applied to: "
    "- Fine hair strands at the edges "
    "- Transparent or sheer fabric (lace, tulle, organza) "
    "- Transparent accessories (glasses, plastic items). "
    "Skin, solid clothing, and the main body must always be solid opaque white. "
    "Strictly exclude and mask out any cast shadows, floor reflections, and detached objects on the floor. "
    "The output should be a clean, noise-free segmentation mask suitable for professional compositing."
)

# 人物用（実写）- V2
# 参考訳：
# 人物のプロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# 【最重要】色と透明度の分離指示：
# 被写体の元の色や影を透明度と混同しないでください。
# 暗い色（衣服の黒いストライプや暗い髪など）は透明ではなく、ソリッドな白でレンダリングする必要があります。
# 絶対的な不透明（白）領域の定義：
# 以下の領域は、内部に灰色のピクセルを一切含まない、完全に均一なソリッドな不透明の白（#FFFFFF）領域としてレンダリングする必要があります：
# 1. 肌の全領域（顔、腕、体）。肌は決して透明ではありません。
# 2. 衣服の全領域（すべてのパターンやストライプを含む）。生地はソリッドです。
# 3. 髪の塊のメインボディ。
# 4. ソリッドなアクセサリーや眼鏡のフレーム。
# 眼鏡の特別ルール：
# 肌の上に位置する透明なレンズは、下の肌が不透明であるため、ソリッドな白でレンダリングする必要があります。
# 半透明（グレー）の限定的な適用：
# エッジの部分でのみ、グレーの階調を使用して半透明を正確に表現してください：
# - 背景に移行する細い個別の髪の毛
# - 本当にシアーな生地（レースなど）が背景と接するエッジ
# 最終出力は、メイン被写体がソリッドな白いシルエットで、グレーがエッジの精細化にのみ使用される、
# クリーンなセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_HUMAN_V2 = (
    "A professional high-precision grayscale alpha matte of the person. "
    "Background must be pure black (#000000). "
    "CRITICAL: Do NOT confuse the subject's original colors or shadows with transparency. "
    "Dark colors (like black stripes on clothes or dark hair) are NOT transparent and must be rendered as solid white. "
    "The following areas must be rendered as a completely uniform, solid opaque white (#FFFFFF) area with absolutely NO gray pixels inside: "
    "1. The entire area of skin (face, arms, body). Skin is never transparent. "
    "2. The entire area of clothing, INCLUDING all patterns and stripes. The fabric is solid. "
    "3. The main body of the hair mass. "
    "4. Solid accessories and frames of glasses. "
    "Special Rule for Glasses: Transparent lenses located ON TOP OF the skin area must be rendered as solid white, because the skin underneath is opaque. "
    "Only use shades of gray for accurately representing semi-transparency at the very edges: "
    "- Fine, individual hair strands transitioning to the background. "
    "- Edges of truly sheer fabrics (like lace) where they meet the background. "
    "The final output must be a clean segmentation mask where the main subject is a solid white silhouette, and gray is used only for edge refinement."
)

# 人物用（実写）- 推奨（汎用版）
# 参考訳：
# 入力画像の人物の、プロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# 【汎用化】色と透明度の分離指示：
# 特定の柄ではなく「暗い色や影、柄」全般を対象にします
# 被写体の元の色、照明の影、暗いテクスチャを透明度と混同しないでください。
# 暗い領域（黒い衣服、暗い髪、深い影、プリントパターンなど）は透明ではなく、
# ソリッドな白でレンダリングする必要があります。
# 絶対的な不透明（白）領域の定義：
# 以下の領域は、内部に灰色のピクセルを一切含まない、完全に均一なソリッドな不透明の白（#FFFFFF）領域としてレンダリングする必要があります：
# 1. 肌の全領域。肌は決して透明ではありません。
# 2. 衣服の全領域（色、パターン、プリント、ロゴに関係なく）。生地自体はソリッドです。
# 3. 髪の塊のメインボディ。
# 4. ソリッドなアクセサリーや眼鏡のフレーム。
# 眼鏡・透過物の汎用ルール：
# 肌や衣服の上に位置する透明なアイテム（眼鏡のレンズなど）は、
# 下のオブジェクトが不透明であるため、ソリッドな白でレンダリングする必要があります。
# 半透明（グレー）の限定的な適用：
# エッジの部分でのみ、グレーの階調を使用して半透明を正確に表現してください：
# - 背景に移行する細い個別の髪の毛
# - 本当にシアーな生地（レースなど）が背景と接するエッジ
# 最終出力は、メイン被写体がソリッドな白いシルエットで、グレーがエッジの精細化にのみ使用される、
# クリーンなセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_HUMAN_GENERIC = (
    "A professional high-precision grayscale alpha matte of the person from the input image. "
    "Background must be pure black (#000000). "
    "CRITICAL: Do NOT confuse the subject's original colors, lighting shadows, or dark textures with transparency. "
    "Dark areas (such as black clothing, dark hair, deep shadows, or printed patterns) are NOT transparent and must be rendered as solid white. "
    "The following areas must be rendered as a completely uniform, solid opaque white (#FFFFFF) area with absolutely NO gray pixels inside: "
    "1. The entire area of skin. Skin is never transparent. "
    "2. The entire area of clothing, regardless of its color, pattern, print, or logo. The fabric itself is solid. "
    "3. The main body of the hair mass. "
    "Only use shades of gray for accurately representing semi-transparency at the very edges: "
    "- Fine, individual hair strands transitioning to the background. "
    "- Edges of truly sheer fabrics (like lace) where they meet the background. "
    "The final output must be a clean segmentation mask where the main subject is a solid white silhouette, and gray is used only for edge refinement."
)


# アルファマットプロンプトマップ（UIドロップダウン用）
ALPHA_MATTE_PROMPTS = {
    "人物用（実写）- 推奨": ALPHA_MATTE_PROMPT_HUMAN_GENERIC,
    "人物用（実写）- V2": ALPHA_MATTE_PROMPT_HUMAN_V2,
    "人物用（実写）- V1": ALPHA_MATTE_PROMPT_HUMAN,
    "キャラクター用（イラスト）": ALPHA_MATTE_PROMPT_CHARACTER,
}
