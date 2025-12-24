import os
from io import BytesIO
from typing import Optional


API_KEY_CANDIDATES = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GENAI_API_KEY",
)


def _first_non_empty(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_local_secrets_key() -> Optional[str]:
    try:
        import local_secrets  # type: ignore
    except Exception:
        return None

    for key_name in API_KEY_CANDIDATES:
        value = getattr(local_secrets, key_name, None)
        key = _first_non_empty(value)
        if key:
            return key
    return None


def _load_streamlit_secrets_toml() -> Optional[str]:
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return None

    try:
        import tomllib
    except Exception:
        return None

    try:
        with open(secrets_path, "rb") as file_handle:
            data = tomllib.load(file_handle)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    for key_name in API_KEY_CANDIDATES:
        key = _first_non_empty(data.get(key_name))
        if key:
            return key
    return None


def get_google_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    key = _first_non_empty(explicit_key)
    if key:
        return key

    key = _load_local_secrets_key()
    if key:
        return key

    key = _load_streamlit_secrets_toml()
    if key:
        return key

    try:
        import streamlit as st  # type: ignore

        for key_name in API_KEY_CANDIDATES:
            if key_name in st.secrets:
                key = _first_non_empty(st.secrets[key_name])
                if key:
                    return key
    except Exception:
        pass

    for key_name in API_KEY_CANDIDATES:
        key = _first_non_empty(os.getenv(key_name))
        if key:
            return key

    return None


def set_proxy(proxy_url: Optional[str]) -> None:
    proxy_url = _first_non_empty(proxy_url)
    if not proxy_url:
        return

    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url


def get_genai_client(api_key: Optional[str] = None):
    from google import genai

    key = get_google_api_key(api_key)
    if not key:
        raise ValueError(
            "Missing Google API key. Set env `GOOGLE_API_KEY`/`GEMINI_API_KEY`, "
            "or create `.streamlit/secrets.toml`, or add `local_secrets.py`."
        )
    return genai.Client(api_key=key)


def configure_google_generativeai(api_key: Optional[str] = None):
    import google.generativeai as genai

    key = get_google_api_key(api_key)
    if not key:
        raise ValueError(
            "Missing Google API key. Set env `GOOGLE_API_KEY`/`GEMINI_API_KEY`, "
            "or create `.streamlit/secrets.toml`, or add `local_secrets.py`."
        )
    genai.configure(api_key=key)
    return genai


def pil_image_to_part(image):
    from google import genai

    image_format = getattr(image, "format", None) or "PNG"
    image_format = str(image_format).upper()
    mime_type = "image/png"
    if image_format in ("JPG", "JPEG"):
        mime_type = "image/jpeg"
    elif image_format == "WEBP":
        mime_type = "image/webp"

    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return genai.types.Part.from_bytes(data=buffer.getvalue(), mime_type=mime_type)
