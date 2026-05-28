# encoding.py

import base64

# =====================================================
# BASE64
# =====================================================

def encode_base64(text):

    return base64.b64encode(text.encode()).decode()

# =====================================================
# CAESAR
# =====================================================

def caesar_cipher(text, shift=3):

    result = ""

    for c in text:

        if c.isalpha():

            base = ord('a') if c.islower() else ord('A')

            result += chr((ord(c) - base + shift) % 26 + base)

        else:

            result += c

    return result