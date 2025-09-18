import re


def str2float(text: str) -> float:
    """Remove uncertainty brackets from strings and return the float."""
    try:
        # Note that the ending ) is sometimes missing. That is why the code has
        # been modified to treat it as optional. Same logic applies to lists.
        return float(re.sub(r"\(.+\)*", "", text))

    except TypeError:
        if isinstance(text, list) and len(text) == 1:
            return float(re.sub(r"\(.+\)*", "", text[0]))

    except ValueError:
        if text.strip() == ".":
            return 0
        raise
    raise ValueError(f"{text!s} cannot be converted to float")


def safe_str2float(value: str, default=None):
    try:
        return str2float(value)
    except ValueError:
        return value if default is None else default
