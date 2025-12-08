# Gemini & Imagen プロンプトベストプラクティス (2025年版)

このドキュメントは、Gemini 3 / Imagen を使用した画像生成において、高品質な結果を得るためのプロンプト作成戦略（Do's and Don'ts）をまとめたものです。

公式ドキュメント (`prompting-strategies.md`, `image-generation.md`, `gemini-3.md`) の内容を集約しています。

---

## 優先すべきこと (Do's) - 20選

画像生成モデル（特にGemini 3 Pro Image / Imagen 3/4）は、高度な言語理解能力を持っています。単なる単語の羅列ではなく、**「文脈」「詳細」「意図」**を伝えることが重要です。

1.  **シーンを物語として説明する (Descriptive Narrative)**
    *   キーワードの羅列ではなく、一貫した文章でシーンを描写してください。
    *   *Bad:* "Dog, park, ball."
    *   *Good:* "A golden retriever leaping joyfully to catch a red frisbee in a sun-drenched park with autumn leaves falling around."

2.  **具体的かつ明確な指示 (Be Specific)**
    *   曖昧さを排除し、何を描くべきかを明確に述べてください。

3.  **フォトリアリスティックな用語の使用 (Photorealistic Terms)**
    *   写真のようなリアリティを求める場合、カメラ用語を使用します。
    *   例: `85mm portrait lens`, `f/1.8 aperture`, `golden hour lighting`, `bokeh background`, `shot on Kodak Portra 400`.

4.  **スタイルと媒体の明示 (Specify Medium & Style)**
    *   どのような形式のアートかを指定します。
    *   例: `Oil painting`, `Watercolor`, `Ukiyo-e`, `Cyberpunk digital art`, `3D render (Octane render)`, `Line art sticker`.

5.  **被写体の詳細化 (Detail the Subject)**
    *   人物の場合: 性別、年齢、人種、髪型、髪色、目の色、服装、表情、ポーズを具体的に定義します。
    *   "A woman" → "A 20-year-old Japanese woman with short bob black hair, wearing a vintage denim jacket, smiling softly."

6.  **照明と雰囲気の設定 (Lighting & Mood)**
    *   光の当たり方やシーンのムードは画像の品質を大きく左右します。
    *   例: `Cinematic lighting`, `Rembrandt lighting`, `Soft morning light`, `Neon backlighting`, `Moody atmosphere`.

7.  **構造化されたプロンプト (Structured Prompting)**
    *   複雑な指示は、XMLタグやMarkdown見出しを使って構造化します。
    *   `<subject>...</subject>`, `<style>...</style>`, `<environment>...</environment>` のように分離するとモデルが理解しやすくなります。

8.  **役割（ペルソナ）の付与 (Assign a Persona)**
    *   システムプロンプトやプロンプトの冒頭で、AIに役割を与えます。
    *   例: "You are an award-winning National Geographic photographer."

9.  **出力形式の指定 (Output Format)**
    *   プロンプト最適化を行う場合、最終的な出力がどのような形式であるべきかを指定します（例: 「英語の記述的な1段落」）。

10. **思考プロセスの導入 (Think Step-by-Step)**
    *   特にGemini 3では、「まずプランを立て、不足要素を特定し、埋めてから生成する」という手順を踏ませることで品質が向上します。

11. **欠落情報の能動的補完 (Fill in the Blanks)**
    *   ユーザーが指定していない要素（背景、時間帯、天候）を「未定義」のままにせず、文脈に合わせてAIに決定させます。

12. **肯定表現の使用 (Positive Phrasing)**
    *   「〜しない」という否定形は無視されやすいため、「〜する」という肯定形で指示します。
    *   *Bad:* "No blurry background."
    *   *Good:* "Sharp focus on the subject with a crystal clear background."（または意図的に "Bokeh background"）

13. **構図の指定 (Composition)**
    *   カメラアングルやフレーミングを指定します。
    *   例: `Close-up`, `Wide angle`, `Overhead shot (Bird's eye view)`, `Low angle`.

14. **高解像度向けの詳細 (High-Res Details)**
    *   4K生成などを目指す場合、ディテールを強調する言葉を含めます。
    *   例: `Highly detailed`, `Intricate textures`, `8k resolution`, `Masterpiece`.

15. **配色の指定 (Color Palette)**
    *   全体の色調を指定します。
    *   例: `Monochromatic`, `Vibrant colors`, `Pastel tones`, `Cyberpunk neon colors`.

16. **テキストレンダリングの指示 (Text Rendering)**
    *   画像内に文字を含める場合、フォントスタイルや配置を具体的に指示します。
    *   例: "With the text 'Coffee Time' written in elegant cursive script on the mug."

17. **参照画像の活用と統合 (Reference Integration)**
    *   複数の参照画像（被写体、スタイル参考など）がある場合、それぞれの役割を明確にします。

18. **多段階の推論 (Multi-step Reasoning)**
    *   「意図の理解」→「詳細の決定」→「プロンプト構築」のステップをプロンプト内で明示します。

19. **自己批判（Self-Correction）**
    *   生成したプロンプトが元の意図から逸脱していないか、バイアスが含まれていないかを確認させます。

20. **英語プロンプトの推奨 (English Prompts)**
    *   画像生成モデルは英語の学習データが圧倒的に多いため、ユーザー入力が日本語であっても、最終的にモデルに渡すプロンプトは英語に変換・最適化することを推奨します。

---

## 避けるべきこと (Don'ts) - 20選

1.  **キーワードの羅列 (Word Soup)**
    *   文脈のない単語リスト（タグクラウド形式）は避けてください。Geminiは文脈を理解するモデルです。

2.  **抽象的すぎる表現 (Vague Terms)**
    *   "Nice image", "Cool style", "Beautiful picture" などの主観的で定義が曖昧な言葉は、意図しない結果を招きます。

3.  **ランダム性の放置 (Leaving it up to chance)**
    *   特定のイメージがある場合、髪型や服の色などを「おまかせ」にせず、具体的に指定してください。指定しないとモデルがランダムに決定し、再生成で一貫性が失われます。

4.  **矛盾する指示 (Conflicting Instructions)**
    *   "Oil painting" と "Photorealistic photo" を同時に指定するなど、スタイルの矛盾を避けてください。

5.  **過度な丁寧語・説得 (Over-politeness)**
    *   "Please", "I would like to..." などの丁寧語は不要です。トークンを消費するだけで、生成品質には寄与しません。直接的に指示してください。

6.  **否定命令の多用 (Negative Constraints)**
    *   "Do not include..." や "No red" などの否定形は、逆にその要素を出現させてしまうことがあります（"red" という単語に注目してしまうため）。

7.  **コンテキストの欠如 (Lack of Context)**
    *   背景情報なしに唐突な指示を出さないでください。「誰が」「どこで」「何を」しているかが最低限必要です。

8.  **構造の混在 (Inconsistent Formatting)**
    *   XMLとMarkdownを混ぜたり、区切り文字がバラバラだったりすると、モデルが構造を正しく解析できない場合があります。

9.  **不要な冗長性 (Unnecessary Verbosity)**
    *   指示自体が長すぎて、肝心の生成内容（Subject）が埋もれないようにしてください。システム指示は簡潔に。

10. **事実性の過信 (Hallucination Risk)**
    *   架空の画像生成において、現実の厳密な事実（特定のマイナーな建物の窓の数など）にこだわりすぎないでください。

11. **思考プロセスの省略 (Skipping Thought Process)**
    *   複雑なシーン構築をワンステップ（入力→即プロンプト出力）で行わせると、詳細が抜け落ちやすくなります。

12. **ユーザー意図の無視 (Ignoring User Intent)**
    *   最適化の過程で、ユーザーが指定した重要な要素（例：特定の髪色）を勝手に変更しないでください。

13. **不可能な要求 (Impossible Requests)**
    *   静止画に対して時間経過（動画的要素）を含めるような指示や、物理的に矛盾する構図（360度全方向を同時に見るなど）。

14. **一貫性のないフォーマット (Inconsistent Few-Shots)**
    *   Few-shot（例示）を与える場合、そのフォーマットは統一してください。

15. **温度パラメータの変更 (Changing Temperature)**
    *   Gemini 3 Pro Image では、デフォルトの `temperature=1.0` が推奨されています。下げすぎると創造性が失われたり、ループしたりする可能性があります。

16. **「何でもいい」という曖昧な出力 (Ambiguous Output)**
    *   最適化の結果として "Something cool" のような曖昧な表現を出力しないでください。具体的な視覚的記述に変換してください。

17. **技術用語の誤用 (Misusing Technical Terms)**
    *   イラストのプロンプトに "ISO 100" などの写真専用用語を混ぜると、画風が崩れる（写真っぽくなる）原因になります。

18. **主語の欠落 (Missing Subject)**
    *   "Running fast in the forest" のように、誰が（何が）主語なのか抜けているプロンプトは避けてください。

19. **複雑すぎる構文 (Overly Complex Syntax)**
    *   モデルが混乱するような、幾重にも入れ子になった文章構造は避けてください。シンプルで力強い文章がベストです。

20. **「思考の署名」の無視 (Ignoring Thought Signatures)**
    *   （開発者向け）マルチターン編集（会話形式での画像編集）を行う際、前回のターンの `thought_signature` を含めないと、コンテキスト（文脈）が失われます。

---

## 3. 推奨プロンプト例 (Good Prompts Examples)

### フォトリアルなポートレート

```
A photorealistic close-up portrait of an elderly Japanese ceramicist with deep, sun-etched wrinkles and a warm, knowing smile.
He is carefully inspecting a freshly glazed tea bowl.
The setting is his rustic, sun-drenched workshop with pottery wheels and shelves of clay pots in the background.
The scene is illuminated by soft, golden hour light streaming through a window, highlighting the fine texture of the clay and the fabric of his apron.
Captured with an 85mm portrait lens at f/1.8, resulting in a soft, blurred background (bokeh).
The overall mood is serene and masterful.
```

### ステッカーデザイン（イラスト）

```
A kawaii-style sticker of a happy red panda wearing a tiny bamboo hat.
It's munching on a green bamboo leaf.
The design features bold, clean outlines, simple cel-shading, and a vibrant color palette.
The background must be white (transparent style).
```

### 商品写真（モックアップ）

```
A high-resolution, studio-lit product photograph of a minimalist ceramic coffee mug in matte black, presented on a polished concrete surface.
The lighting is a three-point softbox setup designed to create soft, diffused highlights and eliminate harsh shadows.
The camera angle is a slightly elevated 45-degree shot to showcase its clean lines.
Ultra-realistic, with sharp focus on the steam rising from the coffee.
Square aspect ratio.
```

---

## 4. プロンプト最適化のためのシステムプロンプト構成案 (System Prompt Strategy)

Gemini 3 の推論能力を活かした、プロンプト最適化エンジンのためのシステムプロンプト構成例です。

```markdown
# Role
You are an expert Art Director and Prompt Engineer for advanced AI image generation (Gemini 3 Pro Image / Imagen 3).

# Task
Transform the user's draft input into a highly detailed, specific, and vivid prompt optimized for image generation models.

# Process (Think step-by-step)
1. **Analyze Intent**:
   - Is the user asking for a Photo, Illustration, 3D Render, or Design?
   - Identify the core subject and action.

2. **Fill in the Blanks (Concrete Decisions)**:
   - Do not leave ambiguous elements. You must make specific artistic choices based on the context.
   - **Subject**: Define vague subjects (e.g., "a girl" -> "a 20yo Japanese woman with short blue hair, wearing a vintage bomber jacket").
   - **Environment**: Define time of day, weather, location details.
   - **Lighting**: Specify the lighting setup (e.g., "Rembrandt lighting", "Neon backlighting").
   - **Technical Specs**:
     - If Photo: Specify Camera, Lens (e.g., "Shot on Sony A7R IV, 35mm lens"), Aperture, Film stock.
     - If Art: Specify Medium, Art Style, Artist references (if appropriate), Line quality.

3. **Construct Narrative**:
   - Combine these decisions into a coherent, descriptive paragraph in English.
   - Ensure the tone matches the desired image style.

# Output Format
Return ONLY the optimized English prompt text.
```
