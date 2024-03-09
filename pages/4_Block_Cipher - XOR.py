import streamlit as st

st.header("Block Cipher - XOR")

def pad(data, block_size):    # CMS (Cryptographic Message Syntax). This pads with the same value as the number of padding bytes.
    # Calculate the number of bytes needed to reach a multiple of block size.
    padding_length = block_size - len(data) % block_size  
    
    # Create the padding by repeating the padding length byte.
    padding = bytes([padding_length] * padding_length)  
    
    # Add the padding to the original data.
    return data + padding                          


def unpad(data):
    # Extract the padding length from the last byte of the data
    padding_length = data[-1]                                # The last byte of the data indicates the length of the padding
    assert padding_length > 0
    message, padding = data[:-padding_length], data[-padding_length:]
    assert all(p == padding_length for p in padding)
    # Remove the padding by slicing the data, excluding the last 'padding_length' bytes
    # This effectively removes the padding from the data
    return message                           # Return the data without the padding


def xor_encrypt_block(plaintext_block, key):
    # Initialize an empty bytes object to store the encrypted block
    encrypted_block = b''
    # Iterate through each byte in the plaintext block
    for i in range(len(plaintext_block)):
        # XOR each byte of the plaintext block with the corresponding byte of the key
        # Use modulus operator to ensure that key bytes are reused if the key length is shorter than the plaintext block length
        encrypted_block += bytes([plaintext_block[i] ^ key[i % len(key)]])
        
    # Return the encrypted block
    return encrypted_block                  


def xor_decrypt_block(ciphertext_block, key):
    return xor_encrypt_block(ciphertext_block, key)  # XOR decryption is same as encryption

def xor_encrypt(plaintext, key, block_size):
    # Initialize an empty bytes object to store the encrypted data
    encrypted_data = b''
    
    # Pad the plaintext to ensure its length is a multiple of the block size
    padded_plaintext = pad(plaintext, block_size)
    
    # Iterate through the plaintext in blocks of size block_size
    st.write("Encrypted blocks")
    for l, i in enumerate(range(0, len(padded_plaintext), block_size)):
        # Extract a block of plaintext
        plaintext_block = padded_plaintext[i:i+block_size]
        # Encrypt the plaintext block using XOR with the key
        encrypted_block = xor_encrypt_block(plaintext_block, key)
        st.write(f"Plain block[{l}]: {plaintext_block.hex()} : {plaintext_block}")
        st.write(f"Cipher block[{l}]: {encrypted_block.hex()} : {encrypted_block}")
        
        # Append the encrypted block to the encrypted data
        encrypted_data += encrypted_block
    # Return the encrypted data
    return encrypted_data                              


def xor_decrypt(ciphertext, key, block_size):
    # Initialize an empty bytes object to store the decrypted data
    decrypted_data = b''
    
    # Iterate through the ciphertext in blocks of size block_size
    st.write("Decrypted blocks")
    for l, i in enumerate(range(0, len(ciphertext), block_size)):
        # Extract the current block of ciphertext
        ciphertext_block = ciphertext[i:i+block_size]
        
        # Decrypt the current block using xor_decrypt_block function
        decrypted_block = xor_decrypt_block(ciphertext_block, key)
        st.write(f"block[{l}]: {decrypted_block.hex()}: {decrypted_block}")
        
        # Append the decrypted block to the decrypted data
        decrypted_data += decrypted_block
    # Remove any padding from the decrypted data
    unpadded_decrypted_data = unpad(decrypted_data)
    
    # Return the unpadded decrypted data
    return unpadded_decrypted_data                              



if __name__ == "__main__":
    # Define the plaintext and encryption key
    allowed_block_sizes = [8, 16, 32, 64, 128]
    
    plaintext = bytes(st.text_input("Plain Text: ").encode())
    key = bytes(st.text_input("Key: ").encode())

    # Define the block size for encryption (adjust according to your needs)

    if st.button("Submit"):
        if not key:
            st.error("Invalid Input!")
        else:
            st.snow()
            while True:
                block_size = int(input(""))
                if block_size in allowed_block_sizes:
                    break
                else:
                    st.write("Block size must be one of 8, 16, 32, 64, or 128 bytes")
                    
            key = pad(key, block_size)   # Pad the key

            # Encryption
            ciphertext = xor_encrypt(plaintext, key, block_size)

            # Decryption
            decrypted_data = xor_decrypt(ciphertext, key, block_size)
            
            st.write("\nOriginal plaintext:", plaintext)
            st.write("Key byte      :", key)
            st.write("Key hex       :", key.hex())
            st.write("Encrypted data:", ciphertext.hex())  # st.write encrypted data in hexadecimal format
            st.write("Decrypted data:", decrypted_data.hex())
            st.write("Decrypted data:", decrypted_data)

# st.write(b'Hello Bob, this '.hex())
