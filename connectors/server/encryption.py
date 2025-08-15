from cryptography.fernet import Fernet

def generate_new_client_key():
    """Generates a key for the Fernet class"""
    key = Fernet.generate_key()
    return key


class Encryption:
    def __init__(self, key):
        self.cipher_suite = Fernet(key)

    def encrypt(self, obj):
        return self.cipher_suite.encrypt(obj)

    def decrypt(self, obj):
        return self.cipher_suite.decrypt(obj)


def demo():
    key = generate_new_client_key()
    cipher = Encryption(key)

    message = b"Hello, this is a secret message."
    msg = cipher.encrypt(message)

    print( cipher.decrypt(msg) )

