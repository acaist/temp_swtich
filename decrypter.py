import os


class Decrypter:
    def __init__(self, keystr:str) -> None:
        self.keystr:str = keystr
    
    def decrypt(self, input_file:str, output_file:str, iscstr=True) -> None:
        print(f"[*] 读取加密文件: {input_file} ...")
        ciphertext = self._readfile(input_file)
        
        print(f"[*] 加载密钥 ...")
        if iscstr:
            key_bytes = self.keystr.encode('utf-8') + b'\x00' # 补齐尾部0x00
        else:
            key_bytes = self.keystr.encode('utf-8')
        
        print(f"[*] 开始解密 ...")
        BOXLEN = 256
        sbox = list(range(BOXLEN))

        # 1. 密钥调度算法 (KSA) - 初始化并混淆 S 盒
        sbox = list(range(BOXLEN))
        key_length = len(key_bytes)
        sumIdx = 0
        for i in range(BOXLEN):
            # 标准 RC4 核心：下标累加严格限制在 0-255，密钥严格循环
            sumIdx = (sumIdx + sbox[i] + key_bytes[i % key_length]) % BOXLEN
            sbox[i], sbox[sumIdx] = sbox[sumIdx], sbox[i]
        
        # 2.对称解密
        i = 0
        sumInx = 0
        plaintext = bytearray()
        for byte in ciphertext:
            i = (i + 1) % BOXLEN
            sumInx = (sumInx + sbox[i]) % BOXLEN
            sbox[i], sbox[sumInx] = sbox[sumInx], sbox[i]
            # 异或解密
            keystream_byte = sbox[(sbox[i] + sbox[sumInx]) % BOXLEN]
            plaintext.append(byte ^ keystream_byte)
    
        print(f"[*] 解密完成, 输出文件...")
        self._writefile(output_file, plaintext)
        
    
    def _readfile(self, input_file:str):
        if not os.path.exists(input_file):
            print(f"[-] 找不到文件路径: {input_file}")
            raise FileNotFoundError()
        
        with open(input_file, "rb") as f:
                ciphertext = f.read()
          
        print(f"[*] 文件读取成功 {int(len(ciphertext)/1000):,} KB")
        return ciphertext

    def _writefile(self, output_file:str, plaintext:bytearray):
        with open(output_file, "wb") as f:
            f.write(plaintext)
    
        print(f"[*] 文件输出至: {output_file}")

def decrypfile(keystr:str, encrypted_target:str, decrypted_output:str="decryptedFile.txt"):
    decrypter = Decrypter(keystr)
    decrypter.decrypt(encrypted_target, decrypted_output)
    
# ====================  ====================
if __name__ =="__main__":
    encrypted_target = r"xxx"
    decrypted_output = r"xxxxxx"

    decrypfile("xxxx", encrypted_target, decrypted_output)
