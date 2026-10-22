import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import compressor_jpeg_sofia as jl

VALORES_S = [0.5, 1, 2, 4]
REPETICOES = 10    # o enunciado pede pelo menos 10
AQUECIMENTO = 2

# pasta onde está o main.py (assim roda certo pelo botão do VS Code também)
pasta_projeto = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) >= 2:
    caminho = sys.argv[1]
else:
    caminho = os.path.join(pasta_projeto, "imagens", "gata.png")

pasta = os.path.join(pasta_projeto, "resultados")
os.makedirs(pasta, exist_ok=True)

if not os.path.exists(caminho):
    print("Imagem não encontrada:", caminho)
    sys.exit()

print("Usando a imagem:", caminho)
print("Aguarde...\n")

# ---------------------------------------------------------------
# Imagem original em tons de cinza (canal Y)
# ---------------------------------------------------------------
Y = jl.ler_imagem_cinza(caminho)
Image.fromarray(Y).save(pasta + "/original_cinza.png")
print("Dimensões da imagem:", Y.shape)

# ---------------------------------------------------------------
# Testes rápidos: matriz C e o exemplo da seção 5.3 do enunciado
# ---------------------------------------------------------------
C = jl.criar_matriz_dct()
print("Erro máximo de C @ C.T - I:", np.abs(C @ C.T - np.eye(8)).max())

exemplo = np.zeros((1, 1, 8, 8), dtype=np.int32)
exemplo[0, 0, 0, 0:3] = [12, -3, 1]
exemplo[0, 0, 1, 0] = 2
vet = jl.zigzag(exemplo, jl.ordem_zigzag(), 1, 1)
cod = jl.rle(vet, 1, 1)
print("RLE do exemplo:", cod[0])
print("Bits do exemplo (esperado 104):", jl.contar_bits(cod))

# ---------------------------------------------------------------
# 9.1 Validação numérica: DCT matricial x DCT rápida
# ---------------------------------------------------------------
blocos, bh, bw = jl.dividir_em_blocos(Y)
F_mat = jl.dct_matricial(blocos, C, bh, bw)
F_fast = jl.dct_rapida(blocos, bh, bw)
print("\ne_max da DCT direta:", np.abs(F_mat - F_fast).max())

f_mat = jl.idct_matricial(F_mat, C, bh, bw)
f_fast = jl.idct_rapida(F_mat, bh, bw)
print("e_max da DCT inversa:", np.abs(f_mat - f_fast).max())

# ---------------------------------------------------------------
# 10. Tempo de execução (só o cálculo da DCT, sem ler arquivo nem plotar)
# ---------------------------------------------------------------
def medir(funcao):
    # roda umas vezes antes sem contar, pra esquentar
    for _ in range(AQUECIMENTO):
        funcao()
    tempos = []
    for _ in range(REPETICOES):
        inicio = time.perf_counter()
        funcao()
        tempos.append(time.perf_counter() - inicio)
    return np.median(tempos)

t_matriz = medir(lambda: jl.dct_matricial(blocos, C, bh, bw))
t_fast = medir(lambda: jl.dct_rapida(blocos, bh, bw))
speedup = t_matriz / t_fast

print("\nt_matriz = %.4f s" % t_matriz)
print("t_fast   = %.4f s" % t_fast)
print("speedup  = %.2f" % speedup)

# ---------------------------------------------------------------
# 8. Curva qualidade x compressão: roda o codec para cada valor de s
# ---------------------------------------------------------------
resultados = []
resultados_fast = []

for s in VALORES_S:
    print("rodando s =", s)
    r = jl.codec(Y, s, usar_rapida=False)
    r_fast = jl.codec(Y, s, usar_rapida=True)
    resultados.append(r)
    resultados_fast.append(r_fast)
    Image.fromarray(r["imagem"]).save(pasta + "/reconstruida_s" + str(s) + ".png")

print("\nVersão A (DCT matricial)")
print("  s |   bpp  | zeros (%) | PSNR (dB) |  SSIM")
for r in resultados:
    print("%4s | %6.3f | %9.2f | %9.2f | %.4f" %
          (r["s"], r["bpp"], r["zeros"], r["psnr"], r["ssim"]))

print("\nVersão B (DCT rápida)")
for r in resultados_fast:
    print("%4s | %6.3f | %9.2f | %9.2f | %.4f" %
          (r["s"], r["bpp"], r["zeros"], r["psnr"], r["ssim"]))

# tabela de síntese (seção 11), salva num txt pra copiar pro relatório
with open(pasta + "/tabela.txt", "w") as arq:
    arq.write("s | bpp | zeros(%) | PSNR | SSIM | t_matriz | t_fast | speedup\n")
    for r in resultados:
        arq.write("%s | %.3f | %.2f | %.2f | %.4f | %.4f | %.4f | %.2f\n" %
                  (r["s"], r["bpp"], r["zeros"], r["psnr"], r["ssim"],
                   t_matriz, t_fast, speedup))

# ---------------------------------------------------------------
# 8. Gráficos PSNR x bpp e SSIM x bpp
# ---------------------------------------------------------------
bpps = [r["bpp"] for r in resultados]
psnrs = [r["psnr"] for r in resultados]
ssims = [r["ssim"] for r in resultados]

plt.figure()
plt.plot(bpps, psnrs, "o-")
plt.xlabel("bpp")
plt.ylabel("PSNR (dB)")
plt.title("PSNR x bpp")
plt.grid(True)
plt.savefig(pasta + "/psnr_bpp.png")

plt.figure()
plt.plot(bpps, ssims, "o-")
plt.xlabel("bpp")
plt.ylabel("SSIM")
plt.title("SSIM x bpp")
plt.grid(True)
plt.savefig(pasta + "/ssim_bpp.png")

# 11. original e reconstruídas lado a lado
plt.figure(figsize=(16, 4))
plt.subplot(1, 5, 1)
plt.imshow(Y, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")
for k, r in enumerate(resultados):
    plt.subplot(1, 5, k + 2)
    plt.imshow(r["imagem"], cmap="gray", vmin=0, vmax=255)
    plt.title("s = " + str(r["s"]))
    plt.axis("off")
plt.savefig(pasta + "/comparacao.png")

print("\nResultados salvos na pasta", pasta)

# mostra os gráficos na tela (fechar as janelas encerra o programa)
plt.show()