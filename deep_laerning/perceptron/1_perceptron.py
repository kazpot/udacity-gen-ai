import numpy as np

def AND(x1, x2):
    # 入力
    x = np.array([x1, x2])
    
    # 重み
    w = np.array([0.5, 0.5])
    
    # バイアス
    b = -0.7

    # 各要素の乗算した後、各要素の和(重み付き和)にバイアスを加算する
    tmp = np.sum(x*w) + b

    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 多層パーセプトロン
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# テスト
print("AND")
print(AND(0, 0))  # 0
print(AND(0, 1))  # 0
print(AND(1, 0))  # 0
print(AND(1, 1))  # 1

# テスト
print("NAND")
print(NAND(0, 0))  # 1
print(NAND(0, 1))  # 1
print(NAND(1, 0))  # 1
print(NAND(1, 1))  # 0

# テスト
print("OR")
print(OR(0, 0))  # 0
print(OR(0, 1))  # 1
print(OR(1, 0))  # 1
print(OR(1, 1))  # 1

# テスト
print("XOR")
print(XOR(0, 0))  # 0
print(XOR(0, 1))  # 1
print(XOR(1, 0))  # 1
print(XOR(1, 1))  # 0