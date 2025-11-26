# Changelog

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
