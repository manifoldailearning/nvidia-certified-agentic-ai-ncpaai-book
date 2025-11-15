# Example snippet from the book - for reference only
import re

def validate_input(user_query: str):
    if len(user_query) > 500:
        raise ValueError("Input too long. Security policy triggered.")
    if re.search(r"(sql|drop|delete|update)", user_query.lower()):
        raise ValueError("Potential command injection detected.")
    return user_query
