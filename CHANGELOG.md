# Changelog

## 2025-11-29: Prompt Optimization Refactoring

### 概要
プロンプト最適化機能を3段階から4段階に拡張し、モデルの使い分けとファイルリスト自動付与機能を実装しました。

### 主な変更内容

#### 1. プロンプト最適化レベルの拡張（3段階 → 4段階）

**新仕様:**
- **Level 0: 無効（整合性チェックのみ）**
  - LLM呼び出しなし
  - 基本プロンプトと追加指示を単純結合
  - 使用モデル: なし

- **Level 1: 修正のみ（誤字脱字・語順）** ⭐ 新規追加
  - 誤字脱字や不自然な語順のみを修正
  - プロンプトの拡張や強化は行わない
  - 使用モデル: `gemini-2.0-flash`（高速・低コスト）

- **Level 2: 標準最適化（推奨）**
  - 旧Level 1相当の機能
  - 自然な統合と最適化
  - 使用モデル: `gemini-3-pro-preview`

- **Level 3: 創造的拡張**
  - 旧Level 2相当の機能
  - 詳細で視覚的に豊かな表現に拡張
  - 使用モデル: `gemini-3-pro-preview`

#### 2. コアファイルの変更: `src/core/prompt_optimizer.py`

**新規メソッド:**
- `_select_model(level: int) -> str`: 最適化レベルに応じてモデルを選択
- `_append_file_list_to_prompt(prompt: str, file_paths: list[str]) -> str`: ファイルリスト情報を自動追加

**メソッド更新:**
- `optimize()`: `file_paths`パラメータを追加（オプショナル）
- システムプロンプトを4段階に対応

**機能追加:**
- Level 1-3で参照画像のファイルリストを自動追記
- Level 1-3で生成日時を自動追記
- 実行順序: LLM最適化 → ファイルリスト追加 → 日時追加

#### 3. UIコンポーネントの更新

全タブのRadioボタンを4段階に更新:
- `src/ui/tabs/basic_edit_tab.py` (lines 925-935)
- `src/ui/tabs/multiturn_edit_tab.py` (lines 810-820)
- `src/ui/tabs/reference_tab.py` (lines 58-68)
- `src/ui/tabs/gemini_tab.py` (lines 52-62)

**変更内容:**
- 選択肢を4つに拡張
- デフォルト値をLevel 1に設定（修正のみ）
- 説明テキストを更新: "レベル1推奨: Gemini 2.0 Flashで誤字脱字を修正"

#### 4. Reference Tabの統合: `src/ui/tabs/reference_tab.py`

**削除:**
- `append_image_info()` メソッド（重複機能のため）

**更新:**
- `generate_optimized_prompt()`: `file_paths`をPromptOptimizerに渡すように変更
- `generate_with_reference_images()`: 同様に`file_paths`を渡すように変更

**効果:**
- コードの重複を削減
- ファイルリスト追加ロジックをPromptOptimizerに一元化

### 影響範囲

**後方互換性:**
- `optimize()`メソッドの`file_paths`パラメータはオプショナル（デフォルト: `None`）
- 既存のコードは変更なしで動作継続

**破壊的変更:**
- UIの選択肢が変更（3段階 → 4段階）
- ユーザーは新しいレベルを選択可能
- デフォルトレベルがLevel 1（修正のみ）に変更

**パフォーマンス:**
- Level 1使用時、Flash APIによりコスト削減・高速化
- Level 2-3は従来通りPro APIを使用

### 技術詳細

**モデル選択ロジック:**
```python
def _select_model(self, level: int) -> str:
    if level == 1:
        return "gemini-2.0-flash"
    elif level in (2, 3):
        return "gemini-3-pro-preview"
    else:
        return "gemini-3-pro-preview"  # フォールバック
```

**ファイルリスト自動付与:**
- 参照画像がある場合、自動的に以下の形式で追記:
```
Input images:
- Input image 1: filename1.jpg
- Input image 2: filename2.png
...
```

**実行順序の保証:**
1. LLMによるプロンプト最適化
2. ファイルリスト情報の追加
3. 日時情報の追加（常に最後）

### テスト推奨項目

- [ ] 各タブでLevel 0-3が正しく選択できる
- [ ] Level 1でFlashモデルが使用される（ログで確認）
- [ ] Level 2-3でProモデルが使用される（ログで確認）
- [ ] Reference Tabで参照画像のファイルリストが自動追記される
- [ ] 日時情報が正しく追記される
- [ ] 後方互換性（既存のAPIコール）が維持される

---

## 2025-11-26: Alpha Processing File Extension Fix

### 問題
Tab2「ベーシック編集」のアルファ処理で生成される中間画像ファイルについて、ファイルの実体は JPG 形式にもかかわらず、ファイル名の拡張子が `.png` になっている問題を修正しました。

### 変更内容

#### 1. 新規ファイル: `src/core/image_utils.py`
画像フォーマット検出のためのユーティリティモジュールを追加:
- `detect_image_format(image_data: bytes) -> str`: バイナリデータから画像フォーマットを検出
- `get_file_extension(image_data: bytes) -> str`: 適切なファイル拡張子を取得（JPEGは`jpg`に正規化）

#### 2. 修正: `samples/gemini_api_sample/nanobanana_edit_alpha_sample.py`
- `get_file_extension()`を使用してアルファマット画像の実際のフォーマットを検出
- 検出されたフォーマットに基づいて正しい拡張子（`.jpg`または`.png`）でファイルを保存
- ドキュメントを更新して、出力ファイルの拡張子が実際のフォーマットに応じて変わることを明記

#### 3. 修正: `samples/gemini_api_sample/nanobanana_edit_alpha_multi_turn.py`
- 同様の修正を適用
- マルチターンチャット版でも正しい拡張子で保存されるように更新

#### 4. 新規テスト: `tests/test_image_utils.py`
包括的なテストを追加:
- JPEG/PNG形式の検出テスト
- ファイル拡張子取得のテスト
- 不正なデータのハンドリングテスト
- すべてのテスト合格

### 影響範囲
- アルファマット中間ファイルが実際のフォーマットと一致する拡張子で保存される
- 拡張子ベースで画像形式を判定するツールやスクリプトが正しく動作する
- 既存のコードとの互換性は維持（画像データ自体は変更なし）

### 技術詳細
- PILの`Image.format`プロパティを使用して画像フォーマットを検出
- エラー時は安全なデフォルト値（`png`）を返す
- JPEG形式は`jpg`拡張子に正規化（`jpeg`ではなく）
