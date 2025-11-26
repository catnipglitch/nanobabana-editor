"""
Image Utilities

画像処理に関するユーティリティ関数を提供するモジュール。
"""

from io import BytesIO
from PIL import Image


def detect_image_format(image_data: bytes) -> str:
    """
    画像バイナリデータから画像フォーマットを検出する。

    Args:
        image_data: 画像のバイナリデータ

    Returns:
        画像フォーマット文字列（小文字）: 'jpeg', 'png', 'webp' など
        検出できない場合は 'png' をデフォルトとして返す

    Examples:
        >>> data = open("image.jpg", "rb").read()
        >>> detect_image_format(data)
        'jpeg'
    """
    try:
        # PILで画像を開いてフォーマットを取得
        img = Image.open(BytesIO(image_data))
        format_str = img.format
        
        if format_str:
            # フォーマット名を小文字に変換
            # JPEG -> jpeg, PNG -> png
            return format_str.lower()
        else:
            # フォーマットが取得できない場合はPNGをデフォルトとする
            return "png"
    except (IOError, OSError):
        # 画像の読み込みやデコードに失敗した場合、PNGをデフォルトとする
        return "png"


def get_file_extension(image_data: bytes) -> str:
    """
    画像バイナリデータから適切なファイル拡張子を取得する。

    Args:
        image_data: 画像のバイナリデータ

    Returns:
        ファイル拡張子（ドットなし）: 'jpg', 'png', 'webp' など

    Examples:
        >>> data = open("image.jpg", "rb").read()
        >>> get_file_extension(data)
        'jpg'
    """
    format_str = detect_image_format(image_data)
    
    # JPEGの場合は 'jpg' に正規化
    if format_str == "jpeg":
        return "jpg"
    
    return format_str
