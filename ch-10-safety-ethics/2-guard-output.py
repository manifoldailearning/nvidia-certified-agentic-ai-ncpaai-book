# Example snippet from the book - for reference only
def guard_output(response: str):
    unsafe_terms = ['violence', 'hate', 'self-harm']
    if any(term in response.lower() for term in unsafe_terms):
        return 'Unsafe response blocked.'
    return response

reply = guard_output('This response contains hate speech')
print(reply)
