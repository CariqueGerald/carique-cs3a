import hashlib
import streamlit as st

st.header("XOR Cipher")

def hash_text(text):
    space_hash = '5C1CE938EC4B836703C845A1D8DB57348758F283'  # Hash value of space
    hashed_chars = set()
    results = []

    # Hash the entire text
    hash_text = hashlib.sha1(text.encode()).hexdigest().upper()

    for char in text:
        if char == ' ':
            if space_hash not in hashed_chars:
                hashed_chars.add(space_hash)
                results.append(f"{space_hash} <space>")
        else:
            hashed_value = hashlib.sha1(char.encode()).hexdigest().upper()
            if hashed_value not in hashed_chars:
                hashed_chars.add(hashed_value)
                results.append(f"{hashed_value} {char}")
    
    results.append(f"{hash_text} {text}")
    return results

text = st.text_input("Enter text to hash:")

if text:
    results = hash_text(text)
    for result in results:
        st.write(result)
