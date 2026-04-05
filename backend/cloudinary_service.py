import os
from typing import Optional

import cloudinary
import cloudinary.uploader


_configured = False


def configure_cloudinary() -> bool:
    global _configured
    if _configured:
        return True

    cloudinary_url = os.environ.get('CLOUDINARY_URL', '').strip()
    if not cloudinary_url:
        return False

    cloudinary.config(cloudinary_url=cloudinary_url, secure=True)
    _configured = True
    return True


def is_cloudinary_enabled() -> bool:
    return configure_cloudinary()


def upload_image(file_path: str, folder: str, public_id: Optional[str] = None) -> dict:
    if not configure_cloudinary():
        raise RuntimeError('Cloudinary is not configured. Set CLOUDINARY_URL.')

    options = {
        'folder': folder,
        'resource_type': 'image',
        'overwrite': False,
    }
    if public_id:
        options['public_id'] = public_id

    result = cloudinary.uploader.upload(file_path, **options)
    secure_url = result.get('secure_url') or result.get('url')
    if not secure_url:
        raise RuntimeError('Cloudinary upload succeeded but did not return an image URL.')
    return result
