"""
Tests for image_utils module
"""

import pytest
from io import BytesIO
from PIL import Image
from src.core.image_utils import detect_image_format, get_file_extension


def test_detect_jpeg_format():
    """JPEG形式の画像を正しく検出できることを確認"""
    # JPEG画像を作成
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    jpeg_data = buffer.getvalue()
    
    # フォーマットを検出
    format_str = detect_image_format(jpeg_data)
    assert format_str == "jpeg"


def test_detect_png_format():
    """PNG形式の画像を正しく検出できることを確認"""
    # PNG画像を作成
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    png_data = buffer.getvalue()
    
    # フォーマットを検出
    format_str = detect_image_format(png_data)
    assert format_str == "png"


def test_get_file_extension_jpeg():
    """JPEG画像から正しいファイル拡張子（jpg）を取得できることを確認"""
    # JPEG画像を作成
    img = Image.new("RGB", (100, 100), color="green")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    jpeg_data = buffer.getvalue()
    
    # 拡張子を取得（jpegではなくjpgが返されることを確認）
    ext = get_file_extension(jpeg_data)
    assert ext == "jpg"


def test_get_file_extension_png():
    """PNG画像から正しいファイル拡張子を取得できることを確認"""
    # PNG画像を作成
    img = Image.new("RGB", (100, 100), color="yellow")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    png_data = buffer.getvalue()
    
    # 拡張子を取得
    ext = get_file_extension(png_data)
    assert ext == "png"


def test_detect_format_with_invalid_data():
    """不正なデータの場合、デフォルト値（png）が返されることを確認"""
    invalid_data = b"This is not an image"
    
    # デフォルトでpngが返される
    format_str = detect_image_format(invalid_data)
    assert format_str == "png"


def test_detect_rgba_png():
    """RGBA PNG画像を正しく検出できることを確認"""
    # RGBA PNG画像を作成
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    png_data = buffer.getvalue()
    
    # フォーマットを検出
    format_str = detect_image_format(png_data)
    assert format_str == "png"
