"""
Tests for PromptOptimizer class
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import MagicMock
from src.core.prompt_optimizer import PromptOptimizer, OPTIMIZATION_SYSTEM_PROMPTS_JA, OPTIMIZATION_SYSTEM_PROMPTS_EN


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def optimizer():
    """PromptOptimizerインスタンスを作成"""
    return PromptOptimizer(api_key="test_api_key")


@pytest.fixture
def mock_genai_client(mocker):
    """Gemini APIクライアントのモック"""
    mock_client = mocker.MagicMock()

    # 正常レスポンスのモック
    mock_response = mocker.MagicMock()
    mock_candidate = mocker.MagicMock()
    mock_content = mocker.MagicMock()
    mock_part = mocker.MagicMock()
    mock_part.text = "Optimized prompt from Gemini"
    mock_content.parts = [mock_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response

    # genai.Clientのモック化
    mocker.patch('google.genai.Client', return_value=mock_client)

    return mock_client


# =============================================================================
# 基本機能テスト (Level 0)
# =============================================================================

def test_level_0_simple_combination(optimizer):
    """Level 0で基本プロンプトとユーザー指示を結合"""
    base = "Generate an image"
    user = "with blue sky"

    prompt, error = optimizer.optimize(base, user, level=0)

    assert "Generate an image" in prompt
    assert "with blue sky" in prompt
    assert error is None


def test_level_0_empty_user_instructions(optimizer):
    """ユーザー指示が空の場合、基本プロンプトのみ返す"""
    base = "Generate an image"
    user = ""

    prompt, error = optimizer.optimize(base, user, level=0)

    assert prompt == base
    assert error is None


def test_level_0_with_file_paths(optimizer):
    """ファイルパスリスト付きLevel 0"""
    base = "Edit the image"
    user = "Make it brighter"
    files = ["/path/to/image1.jpg", "/path/to/image2.png"]

    prompt, error = optimizer.optimize(base, user, level=0, file_paths=files)

    assert "Edit the image" in prompt
    assert "Make it brighter" in prompt
    assert "Input images:" in prompt
    assert "image1.jpg" in prompt
    assert "image2.png" in prompt
    assert error is None


# =============================================================================
# ファイル名処理テスト
# =============================================================================

def test_append_file_list_single(optimizer):
    """単一ファイル追加"""
    prompt = "Test prompt"
    files = ["/path/to/cat.jpg"]

    result = optimizer._append_file_list_to_prompt(prompt, files)

    assert "Input images:" in result
    assert "Input image 1: cat.jpg" in result


def test_append_file_list_multiple(optimizer):
    """複数ファイル追加"""
    prompt = "Test prompt"
    files = ["/path/to/cat.jpg", "/another/path/dog.png", "/images/bird.webp"]

    result = optimizer._append_file_list_to_prompt(prompt, files)

    assert "Input images:" in result
    assert "Input image 1: cat.jpg" in result
    assert "Input image 2: dog.png" in result
    assert "Input image 3: bird.webp" in result


def test_append_file_list_empty(optimizer):
    """空リストの場合、何も追加しない"""
    prompt = "Test prompt"
    files = []

    result = optimizer._append_file_list_to_prompt(prompt, files)

    assert result == prompt
    assert "Input images:" not in result


def test_append_file_list_duplicate_prevention(optimizer):
    """重複追加防止"""
    prompt = "Test prompt\n\nInput images:\n- Input image 1: existing.jpg"
    files = ["/path/to/new.jpg"]

    result = optimizer._append_file_list_to_prompt(prompt, files)

    # 既存の "Input images:" が含まれているため、追加されない
    assert result == prompt


def test_append_file_list_filename_extraction(optimizer):
    """パスからファイル名を正しく抽出"""
    prompt = "Test prompt"
    files = ["/very/long/path/to/some/image.jpg"]

    result = optimizer._append_file_list_to_prompt(prompt, files)

    assert "image.jpg" in result
    assert "/very/long/path" not in result


# =============================================================================
# 最適化レベルテスト (モック使用)
# =============================================================================

def test_level_1_calls_flash_model(optimizer, mock_genai_client):
    """Level 1が gemini-2.0-flash を使用"""
    base = "Test prompt"
    user = "Test instructions"

    prompt, error = optimizer.optimize(base, user, level=1)

    # モデル選択を確認
    assert optimizer._select_model(1) == "gemini-2.0-flash"
    assert error is None


def test_level_2_calls_pro_model(optimizer, mock_genai_client):
    """Level 2が gemini-3-pro-preview を使用"""
    base = "Test prompt"
    user = "Test instructions"

    prompt, error = optimizer.optimize(base, user, level=2)

    # モデル選択を確認
    assert optimizer._select_model(2) == "gemini-3-pro-preview"
    assert error is None


def test_level_3_calls_pro_model(optimizer, mock_genai_client):
    """Level 3が gemini-3-pro-preview を使用"""
    base = "Test prompt"
    user = "Test instructions"

    prompt, error = optimizer.optimize(base, user, level=3)

    # モデル選択を確認
    assert optimizer._select_model(3) == "gemini-3-pro-preview"
    assert error is None


def test_optimization_with_file_paths(optimizer, mock_genai_client):
    """ファイルパス付き最適化"""
    base = "Edit image"
    user = "Apply filter"
    files = ["/path/to/image.jpg"]

    prompt, error = optimizer.optimize(base, user, level=2, file_paths=files)

    assert "Optimized prompt from Gemini" in prompt
    assert "Input images:" in prompt
    assert "image.jpg" in prompt
    assert error is None


# =============================================================================
# エラー処理テスト
# =============================================================================

def test_fallback_on_empty_response(optimizer, mocker):
    """空レスポンス時のフォールバック"""
    # 空レスポンスを返すモック
    mock_client = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.candidates = []
    mock_client.models.generate_content.return_value = mock_response
    mocker.patch('google.genai.Client', return_value=mock_client)

    base = "Test prompt"
    user = "Test instructions"

    prompt, error = optimizer.optimize(base, user, level=2)

    # フォールバックプロンプトが返されることを確認
    assert "Test prompt" in prompt
    assert "Test instructions" in prompt
    assert error is None


def test_fallback_on_api_error(optimizer, mocker):
    """API例外時のフォールバック"""
    # 例外を発生させるモック
    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Error")
    mocker.patch('google.genai.Client', return_value=mock_client)

    base = "Test prompt"
    user = "Test instructions"

    prompt, error = optimizer.optimize(base, user, level=2)

    # フォールバックプロンプトが返されることを確認
    assert "Test prompt" in prompt
    assert "Test instructions" in prompt
    # エラーメッセージが返されることを確認
    assert error is not None
    assert "プロンプト最適化エラー" in error


def test_fallback_preserves_file_paths(optimizer, mocker):
    """フォールバック時もファイルリストを保持"""
    # 例外を発生させるモック
    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Error")
    mocker.patch('google.genai.Client', return_value=mock_client)

    base = "Edit image"
    user = "Apply filter"
    files = ["/path/to/image.jpg"]

    prompt, error = optimizer.optimize(base, user, level=2, file_paths=files)

    # ファイルリストが含まれていることを確認
    assert "Input images:" in prompt
    assert "image.jpg" in prompt
    assert error is not None


# =============================================================================
# システムプロンプトテスト
# =============================================================================

def test_get_system_prompt_level_0(optimizer):
    """Level 0は空文字列"""
    prompt = optimizer._get_system_prompt(0)
    assert prompt == ""


def test_get_system_prompt_level_1(optimizer):
    """Level 1のプロンプト取得"""
    prompt = optimizer._get_system_prompt(1)
    assert len(prompt) > 0
    assert "correction" in prompt.lower() or "修正" in prompt


def test_get_system_prompt_level_2(optimizer):
    """Level 2のプロンプト取得"""
    prompt = optimizer._get_system_prompt(2)
    assert len(prompt) > 0
    assert "optimize" in prompt.lower() or "最適化" in prompt


def test_get_system_prompt_level_3(optimizer):
    """Level 3のプロンプト取得"""
    prompt = optimizer._get_system_prompt(3)
    assert len(prompt) > 0
    assert ("creative" in prompt.lower() or "vivid" in prompt.lower() or
            "創造" in prompt or "鮮明" in prompt)


def test_system_prompt_language_filtering_en(optimizer):
    """英語モードでコメント行が含まれていないことを確認"""
    # 英語モードでプロンプトを取得
    prompt = optimizer._get_system_prompt(1, language="en")

    # コメント行（#で始まる）が含まれていないことを確認
    lines = prompt.splitlines()
    for line in lines:
        assert not line.strip().startswith("#")


def test_system_prompt_contains_temporal_instruction_level_2(optimizer):
    """Level 2のシステムプロンプトに時間処理指示が含まれる"""
    prompt = optimizer._get_system_prompt(2)

    # 時間処理の指示が含まれていることを確認
    assert "temporal references" in prompt.lower() or "時間参照" in prompt


def test_system_prompt_contains_temporal_instruction_level_3(optimizer):
    """Level 3のシステムプロンプトに時間処理指示が含まれる"""
    prompt = optimizer._get_system_prompt(3)

    # 時間処理の指示が含まれていることを確認
    # Note: Terminology updated to "temporal refs" / "時勢表現" in newer prompt versions
    assert "temporal refs" in prompt.lower() or "時勢表現" in prompt


def test_system_prompt_contains_negative_constraint_instruction_level_3(optimizer):
    """Level 3のシステムプロンプトに否定制約の維持指示が含まれる"""
    prompt = optimizer._get_system_prompt(3)

    # 否定制約（No text等）の維持指示が含まれていることを確認
    assert "no text" in prompt.lower()
    assert "no logos" in prompt.lower()
    assert "negative constraints" in prompt.lower() or "否定形の扱い" in prompt


def test_system_prompt_contains_character_defaults_level_3(optimizer):
    """Level 3のシステムプロンプトに人物のデフォルト指示（着こなし、表情）が含まれる"""
    prompt = optimizer._get_system_prompt(3)

    # 日本語キーワードのチェック
    if "人物のデフォルト" in prompt:
        assert "着こなし" in prompt or "服装は正しく" in prompt
        assert "ポジティブ" in prompt or "positive" in prompt
    # 英語キーワードのチェック (Character Defaults)
    elif "Character Defaults" in prompt:
        assert "Attire must be worn correctly" in prompt
        assert "positive" in prompt


# =============================================================================
# ヘルパーメソッドテスト
# =============================================================================

def test_create_fallback_prompt_basic(optimizer):
    """_create_fallback_promptの基本動作"""
    base = "Generate image"
    user = "with mountains"

    result = optimizer._create_fallback_prompt(base, user)

    assert "Generate image" in result
    assert "with mountains" in result


def test_create_fallback_prompt_with_files(optimizer):
    """ファイルパス付きフォールバック"""
    base = "Edit image"
    user = "Apply effect"
    files = ["/path/to/test.jpg"]

    result = optimizer._create_fallback_prompt(base, user, files)

    assert "Edit image" in result
    assert "Apply effect" in result
    assert "Input images:" in result
    assert "test.jpg" in result


def test_level_0_consistency_check(optimizer):
    """_level_0_consistency_checkの単独テスト"""
    base = "Base prompt"
    user = "User instructions"

    result, error = optimizer._level_0_consistency_check(base, user)

    assert "Base prompt" in result
    assert "User instructions" in result
    assert error is None


def test_level_0_consistency_check_empty_user(optimizer):
    """ユーザー指示が空の場合"""
    base = "Base prompt"
    user = ""

    result, error = optimizer._level_0_consistency_check(base, user)

    assert result == base
    assert error is None


# =============================================================================
# モデル選択テスト
# =============================================================================

def test_select_model_level_1(optimizer):
    """Level 1はFlashモデル"""
    assert optimizer._select_model(1) == "gemini-2.0-flash"


def test_select_model_level_2(optimizer):
    """Level 2はProモデル"""
    assert optimizer._select_model(2) == "gemini-3-pro-preview"


def test_select_model_level_3(optimizer):
    """Level 3はProモデル"""
    assert optimizer._select_model(3) == "gemini-3-pro-preview"


def test_select_model_invalid_level(optimizer):
    """無効なレベルはProモデルにフォールバック"""
    assert optimizer._select_model(99) == "gemini-3-pro-preview"


# =============================================================================
# 統合テスト
# =============================================================================

def test_full_optimization_workflow(optimizer, mock_genai_client):
    """完全な最適化ワークフロー"""
    base = "Generate portrait"
    user = "Professional lighting"
    files = ["/path/to/reference.jpg"]

    prompt, error = optimizer.optimize(base, user, level=2, file_paths=files)

    # 最適化されたプロンプトが返されることを確認
    assert "Optimized prompt from Gemini" in prompt
    # ファイルリストが追加されることを確認
    assert "Input images:" in prompt
    assert "reference.jpg" in prompt
    # エラーがないことを確認
    assert error is None


def test_no_datetime_in_prompt(optimizer):
    """プロンプトに "Current datetime:" が含まれないことを確認（時勢処理削除後）"""
    base = "Generate image for tomorrow"
    user = "with sunrise"

    prompt, error = optimizer.optimize(base, user, level=0)

    # 日時情報が追加されていないことを確認
    assert "Current datetime:" not in prompt


def test_datetime_context_in_level_1_prompt(optimizer, mock_genai_client):
    """Level 1で日時コンテキストがLLMに渡されることを確認"""
    base = "Generate image for tomorrow"
    user = "with sunrise"

    prompt, error = optimizer.optimize(base, user, level=1)

    call_args = mock_genai_client.models.generate_content.call_args
    sent_prompt = call_args.kwargs['contents']

    assert "Current datetime:" in sent_prompt
    assert "<datetime_context>" in sent_prompt
    assert error is None


def test_datetime_context_in_level_2_prompt(optimizer, mock_genai_client):
    """Level 2で日時コンテキストがLLMに渡されることを確認"""
    base = "Generate image for next week"
    user = "with mountains"

    prompt, error = optimizer.optimize(base, user, level=2)

    call_args = mock_genai_client.models.generate_content.call_args
    sent_prompt = call_args.kwargs['contents']

    assert "Current datetime:" in sent_prompt
    assert "<datetime_context>" in sent_prompt
    assert error is None


def test_datetime_context_format(optimizer):
    """日時コンテキストのフォーマットを確認"""
    context = optimizer._get_datetime_context()

    assert "<datetime_context>" in context
    assert "Current datetime:" in context
    assert "</datetime_context>" in context

    # 日時フォーマットの検証（YYYY-MM-DD HH:MM:SS）
    import re
    datetime_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    assert re.search(datetime_pattern, context) is not None
