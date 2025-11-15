import re

def mask_pii(text):
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-z.-]+\.[a-z]{2,}", "[EMAIL]", text)
    return text

# Example usage
text = "My phone number is 1234567890 and my email is test@example.com"
masked = mask_pii(text)
print(masked)