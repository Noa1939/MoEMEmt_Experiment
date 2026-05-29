import base64

# =====================================================
# BASE64
# =====================================================

def encode_base64(text):

    return base64.b64encode(
        text.encode("utf-8")
    ).decode("utf-8")


def decode_base64(text):

    return base64.b64decode(
        text
    ).decode("utf-8")


# =====================================================
# CAESAR
# =====================================================

def caesar_cipher(text, shift=3):

    result = ""

    for c in text:

        if c.isalpha():

            base = (
                ord("a")
                if c.islower()
                else ord("A")
            )

            result += chr(
                (ord(c) - base + shift) % 26 + base
            )

        else:

            result += c

    return result


def decode_caesar(text):

    return caesar_cipher(
        text,
        shift=-3
    )