import math
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from skimage.metrics import structural_similarity

# Tabela de quantização de luminância do JPEG (a do enunciado)
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

# Matriz da equação (1) do enunciado (RGB -> YCrCb)
MATRIZ_YCRCB = np.array([
    [0.299, 0.587, 0.114],
    [0.500, -0.418688, -0.081312],
    [-0.168736, -0.331264, 0.500]
])
DESLOCAMENTO = np.array([0, 128, 128])


# ===============================================================
# 1. Requisitos de entrada e preparação
# ===============================================================

# 1.1 Entrada da imagem e conversão RGB -> YCrCb
def ler_imagem_cinza(caminho):
    # o PIL já abre na ordem R, G, B então não precisa trocar canal
    img = Image.open(caminho).convert("RGB")
    rgb = np.array(img, dtype=np.float64)

    # conversão feita na mão com a matriz, sem função pronta
    ycrcb = rgb @ MATRIZ_YCRCB.T + DESLOCAMENTO

    # só o canal Y interessa (é a imagem em cinza)
    Y = np.clip(np.round(ycrcb[:, :, 0]), 0, 255).astype(np.uint8)
    return Y


def dividir_em_blocos(Y):
    # 1.2 Deslocamento de nível
    # precisa ser float, em uint8 o -128 estoura
    deslocada = Y.astype(np.float64) - 128

    # 1.3 Divisão em blocos 8x8
    altura, largura = deslocada.shape

    # quanto falta pra chegar no próximo múltiplo de 8
    falta_altura = (8 - altura % 8) % 8
    falta_largura = (8 - largura % 8) % 8
    padded = np.pad(deslocada, ((0, falta_altura), (0, falta_largura)), mode="edge")

    blocos_h = padded.shape[0] // 8
    blocos_w = padded.shape[1] // 8

    # no final fica (blocos_h, blocos_w, 8, 8)
    blocos = padded.reshape(blocos_h, 8, blocos_w, 8).transpose(0, 2, 1, 3)
    return blocos, blocos_h, blocos_w


# 6.2 (parte final) junta os blocos de volta e tira o padding
def juntar_blocos(blocos, blocos_h, blocos_w, altura, largura):
    imagem = np.zeros((blocos_h * 8, blocos_w * 8))
    for i in range(blocos_h):
        for j in range(blocos_w):
            imagem[i*8:(i+1)*8, j*8:(j+1)*8] = blocos[i, j]

    return imagem[:altura, :largura]


# ===============================================================
# 2. Versão A - DCT 2D pela operação matricial (ENTREGÁVEL 2)
# ===============================================================

# 2.1 Construção da matriz da DCT (só é criada uma vez, no codec)
def criar_matriz_dct(N=8):
    C = np.zeros((N, N))
    for k in range(N):
        # alfa só depende de k
        if k == 0:
            alfa = math.sqrt(1 / N)
        else:
            alfa = math.sqrt(2 / N)
        for n in range(N):
            C[k, n] = alfa * math.cos((math.pi / N) * (n + 0.5) * k)
    return C


# 2.2 DCT 2D, F = C f C^T
def dct_matricial(blocos, C, blocos_h, blocos_w):
    F = np.zeros_like(blocos)
    for i in range(blocos_h):
        for j in range(blocos_w):
            F[i, j] = C @ blocos[i, j] @ C.T
    return F


# 6.2 IDCT matricial, f = C^T F C
def idct_matricial(F, C, blocos_h, blocos_w):
    f = np.zeros_like(F)
    for i in range(blocos_h):
        for j in range(blocos_w):
            f[i, j] = C.T @ F[i, j] @ C
    return f


# ===============================================================
# 9. Versão B - DCT rápida por biblioteca (ENTREGÁVEL 3)
# ===============================================================

# usa norm="ortho" pra ficar com a mesma normalização da matricial
def dct_rapida(blocos, blocos_h, blocos_w):
    F = np.zeros_like(blocos)
    for i in range(blocos_h):
        for j in range(blocos_w):
            F[i, j] = dctn(blocos[i, j], type=2, norm="ortho")
    return F


# inversa da versão B
def idct_rapida(F, blocos_h, blocos_w):
    f = np.zeros_like(F)
    for i in range(blocos_h):
        for j in range(blocos_w):
            f[i, j] = idctn(F[i, j], type=2, norm="ortho")
    return f


# ===============================================================
# 3. Quantização
# ===============================================================

# Fq = round(F / (s * Q)), o numpy divide todos os blocos de uma vez
def quantizar(F, s):
    Qs = s * Q
    return np.round(F / Qs).astype(np.int32)


# 6.1 Desquantização, F = Fq * (s * Q)
def desquantizar(Fq, s):
    Qs = s * Q
    return Fq * Qs


# ===============================================================
# 4. Varredura zig-zag
# ===============================================================

# monta a lista de posições na ordem do zig-zag: (0,0), (0,1), (1,0), (2,0)...
def ordem_zigzag(N=8):
    ordem = []
    for diagonal in range(2 * N - 1):
        linha_min = max(0, diagonal - (N - 1))
        linha_max = min(N - 1, diagonal)

        # diagonal par sobe, diagonal ímpar desce
        if diagonal % 2 == 0:
            linhas = range(linha_max, linha_min - 1, -1)
        else:
            linhas = range(linha_min, linha_max + 1)

        for linha in linhas:
            ordem.append((linha, diagonal - linha))
    return ordem


# bloco 8x8 -> vetor com 64 posições
def zigzag(Fq, ordem, blocos_h, blocos_w):
    vetores = np.zeros((blocos_h, blocos_w, 64), dtype=np.int32)
    for i in range(blocos_h):
        for j in range(blocos_w):
            for k, (linha, coluna) in enumerate(ordem):
                vetores[i, j, k] = Fq[i, j, linha, coluna]
    return vetores


# 6. (inverso) vetor com 64 posições -> bloco 8x8
def zigzag_inverso(vetores, ordem, blocos_h, blocos_w):
    Fq = np.zeros((blocos_h, blocos_w, 8, 8), dtype=np.int32)
    for i in range(blocos_h):
        for j in range(blocos_w):
            for k, (linha, coluna) in enumerate(ordem):
                Fq[i, j, linha, coluna] = vetores[i, j, k]
    return Fq


# ===============================================================
# 5. Codificação simples (RLE) e cálculo do bpp
# ===============================================================

# 5.1 RLE mínimo: pares (run, valor) e um "EOB" no fim de cada bloco
def rle(vetores, blocos_h, blocos_w):
    resultado = []
    for i in range(blocos_h):
        for j in range(blocos_w):
            run = 0
            codigo = []
            for valor in vetores[i, j]:
                if valor == 0:
                    run += 1
                else:
                    codigo.append((run, int(valor)))
                    run = 0
            codigo.append("EOB")
            resultado.append(codigo)
    return resultado


# 5.2 Bits por pixel
# cada par gasta 24 bits (8 do run + 16 do valor) e cada EOB gasta 8.
# o bpp em si é calculado no codec(), dividindo por altura * largura
def contar_bits(rle_blocos):
    total = 0
    for codigo in rle_blocos:
        pares = len(codigo) - 1   # menos o EOB
        total += pares * 24 + 8
    return total


# 6. (inverso) Decodificação RLE
def rle_inverso(rle_blocos, blocos_h, blocos_w):
    vetores = np.zeros((blocos_h, blocos_w, 64), dtype=np.int32)
    indice = 0
    for i in range(blocos_h):
        for j in range(blocos_w):
            posicao = 0
            for item in rle_blocos[indice]:
                if item == "EOB":
                    break
                run, valor = item
                posicao += run   # pula os zeros
                vetores[i, j, posicao] = valor
                posicao += 1
            indice += 1
    return vetores


# ===============================================================
# 7. Métricas de qualidade
# ===============================================================

# 7.1 PSNR (feito direto, sem função pronta)
def calcular_psnr(original, reconstruida):
    diferenca = original.astype(np.float64) - reconstruida.astype(np.float64)
    mse = np.mean(diferenca ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)


# 7.2 SSIM (esse pode usar a função pronta do skimage)
def calcular_ssim(original, reconstruida):
    return structural_similarity(original, reconstruida, data_range=255)


# ===============================================================
# Codec completo (ENTREGÁVEL 1)
# Faz a compressão e a reconstrução para um valor de s.
#   usar_rapida=False -> Versão A (matricial)
#   usar_rapida=True  -> Versão B (DCT rápida, ENTREGÁVEL 3)
# ===============================================================

def codec(Y, s, usar_rapida=False):
    altura, largura = Y.shape
    C = criar_matriz_dct()

    # ida (seções 1 a 5): blocos -> DCT -> quantização -> zig-zag -> RLE
    blocos, blocos_h, blocos_w = dividir_em_blocos(Y)

    if usar_rapida:
        F = dct_rapida(blocos, blocos_h, blocos_w)
    else:
        F = dct_matricial(blocos, C, blocos_h, blocos_w)

    Fq = quantizar(F, s)
    ordem = ordem_zigzag()
    vetores = zigzag(Fq, ordem, blocos_h, blocos_w)
    rle_blocos = rle(vetores, blocos_h, blocos_w)

    # 5.2 taxa: bpp com as dimensões originais (sem o padding)
    total_bits = contar_bits(rle_blocos)
    bpp = total_bits / (altura * largura)

    # % de coeficientes zero (tabela da seção 8)
    percentual_zeros = 100 * np.count_nonzero(Fq == 0) / Fq.size

    # volta (seção 6): RLE^-1 -> zig-zag^-1 -> desquantização -> IDCT
    vetores_rec = rle_inverso(rle_blocos, blocos_h, blocos_w)
    Fq_rec = zigzag_inverso(vetores_rec, ordem, blocos_h, blocos_w)
    F_rec = desquantizar(Fq_rec, s)

    if usar_rapida:
        blocos_rec = idct_rapida(F_rec, blocos_h, blocos_w)
    else:
        blocos_rec = idct_matricial(F_rec, C, blocos_h, blocos_w)

    # 6.2 junta os blocos, tira o padding, soma 128 e limita em [0, 255]
    imagem = juntar_blocos(blocos_rec, blocos_h, blocos_w, altura, largura)
    imagem = imagem + 128
    Y_rec = np.clip(np.round(imagem), 0, 255).astype(np.uint8)

    return {
        "s": s,
        "bpp": bpp,
        "zeros": percentual_zeros,
        "psnr": calcular_psnr(Y, Y_rec),
        "ssim": calcular_ssim(Y, Y_rec),
        "imagem": Y_rec
    }