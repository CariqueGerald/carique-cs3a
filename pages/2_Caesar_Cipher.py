import streamlit as st

st.header("Caesar Cipher")

def encrypt_decrypt(text, shift_keys, ifdecrypt):
    """
    Encrypts a text using Caesar Cipher with a list of shift keys.
    Args:
        text: The text to encrypt.
        shift_keys: A list of integers representing the shift values for each character.
        ifdecrypt: flag if decrypt or encrypt
    Returns:
        A string containing the encrypted text if encrypt and plain text if decrypt
    """
    
    result = []
    key = []
    letters = list(text)
    shiftkeys = shift_keys.split()
    
    for i in range(len(text)):
        key += (shiftkeys[i % len(shiftkeys)]).split()
    
    for i in range(len(text)):
        if ifdecrypt: result += chr((ord(letters[i]) - int(key[i]) - 32) % 94 + 32) 
        else: result += chr((ord(letters[i]) + int(key[i]) - 32 + 94) % 94 + 32)
        st.write(i, letters[i], key[i], result[i])
    st.write("----------")
    
    for i in range(len(text)):
        st.write(i, result[i], key[i], letters[i])
        outcome = "".join(result)
    st.write("----------")
    
    return outcome

# Example usage
text = st.text_input("Enter Text:")
shift_keys = st.text_input("Enter Shift Keys:")

if st.button("Submit"):
    if not shift_keys:
        st.error("Invalid Input!")
    else:
        st.snow()
        x = encrypt_decrypt(text, shift_keys, ifdecrypt = False)

        st.write("Text:", text)
        st.write("Shift keys:", shift_keys)
        st.write("Cipher:", x)
        st.write("Decrypted text:", text)
