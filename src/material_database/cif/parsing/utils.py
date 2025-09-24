import re


def str2float(text: list[str] | str) -> float:
    """Convert a string (or single-element list) to a float, removing uncertainty brackets and handling common issues."""

    # If input is a single-element list, unpack it
    if isinstance(text, list):
        if len(text) == 1:
            text = text[0]

        else:
            raise ValueError("Input list must contain exactly one element.")

    if not isinstance(text, str):
        raise ValueError(
            "Input must be a string or a single-element list containing a string."
        )

    # Convert non-string types to string
    text = text.strip()

    # Handle empty or placeholder strings
    if text in {"", ".", "-", "nan"}:
        return 0.0

    # Remove uncertainty brackets
    text = re.sub(r"\(.+?\)", "", text).strip()

    # Replace colon with dot (optional, depending on your use case)
    if ":" in text:
        text = text.replace(":", ".")

    # Remove commas (thousands separators)
    text = text.replace(",", "")

    if text.count(".") > 1:
        # Keep the first dot, remove subsequent dots
        first_dot_index = text.find(".")
        text = text[: first_dot_index + 1] + text[first_dot_index + 1 :].replace(
            ".", ""
        )

    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{text!s} cannot be converted to float")


def safe_str2float(value: str, default=None):
    try:
        return str2float(value)
    except ValueError:
        return value if default is None else default
