# Compressor de Imagens JPEG-like em Python

Projeto 1VA da disciplina **14710 - Arquiteturas Não-Convencionais** (UFRPE), Prof. Vítor A. Coutinho.

## Sobre o projeto

Este projeto implementa, em Python, um compressor de imagens inspirado no JPEG. A imagem colorida é convertida para tons de cinza, dividida em blocos de 8x8 pixels, transformada com a DCT, quantizada e codificada com um RLE simples. Depois o caminho inverso reconstrói a imagem, e a qualidade é medida com PSNR e SSIM.

O trabalho também compara duas formas de calcular a DCT:

- **Versão A:** DCT matricial implementada à mão (`F = C f Cᵀ`).
- **Versão B:** DCT rápida da biblioteca SciPy (`scipy.fft.dctn`).

## O que foi pedido

| Seção do enunciado | O que fazer |
|---|---|
| 1. Entrada e preparação | Converter RGB para YCrCb com a matriz (sem função pronta), usar o canal Y, subtrair 128 e dividir em blocos 8x8 com padding |
| 2. Versão A | DCT 2D matricial com a matriz C construída uma vez |
| 3. Quantização | Dividir pela tabela Q do JPEG multiplicada pela escala `s` |
| 4. Zig-zag | Reordenar cada bloco em um vetor de 64 posições |
| 5. RLE e bpp | Pares (run, valor) de 24 bits, EOB de 8 bits e cálculo de bits por pixel |
| 6. Procedimento inverso | RLE⁻¹, zig-zag⁻¹, desquantização, IDCT, +128 e remoção do padding |
| 7. Métricas | PSNR (implementado à mão) e SSIM |
| 8. Qualidade x compressão | Rodar com `s` = 0.5, 1, 2 e 4, com tabela e gráficos PSNR x bpp e SSIM x bpp |
| 9. Versão B | Mesmo codec com DCT rápida de biblioteca e validação numérica contra a matricial |
| 10. Tempo | Medir só o cálculo da DCT (mediana de pelo menos 10 repetições) e calcular o speedup |
| 11. Resultados | Imagens, tabelas, gráficos e discussão no relatório |
| 12. Entregáveis | Código, DCT matricial, segunda versão com DCT rápida, relatório e imagens |

## Arquivos do projeto

```
compressor_jpeg_sofia/
├── compressor_jpeg_sofia.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── imagens/
│   └── gata.png
└── resultados/        (criada ao rodar o main.py)
```

| Arquivo | Conteúdo |
|---|---|
| `compressor_jpeg_sofia.py` | Todo o codec: leitura e conversão da imagem, divisão em blocos, DCT matricial (Versão A), DCT rápida (Versão B), quantização, zig-zag, RLE, decodificação, PSNR e SSIM. A função `codec()` roda a compressão e a reconstrução para um valor de `s`. Os comentários indicam a seção do enunciado de cada trecho e onde estão os entregáveis 1, 2 e 3. |
| `main.py` | Executa os experimentos: testes rápidos, validação numérica das duas DCTs, medição de tempo e speedup, tabela por valor de `s`, gráficos e imagens reconstruídas. |
| `requirements.txt` | Bibliotecas necessárias. |
| `imagens/gata.png` | Imagem usada nos testes (foto autoral). |
| `resultados/` | Saídas geradas pelo `main.py`. |

Arquivos gerados em `resultados/`:

- `original_cinza.png`: imagem original em tons de cinza
- `reconstruida_s0.5.png`, `reconstruida_s1.png`, `reconstruida_s2.png`, `reconstruida_s4.png`
- `psnr_bpp.png` e `ssim_bpp.png`: gráficos de taxa e distorção
- `comparacao.png`: original e reconstruídas lado a lado
- `tabela.txt`: tabela de síntese (s, bpp, % zeros, PSNR, SSIM, tempos e speedup)

## Como executar

1. Instale as bibliotecas:

```
pip install -r requirements.txt
```

2. Rode o programa (usa `imagens/gata.png` por padrão):

```
python main.py
```

Para usar outra imagem:

```
python main.py imagens/outra_foto.png
```

Os resultados aparecem no terminal e na pasta `resultados/`. Os gráficos também abrem em janelas no final. O programa pode levar 1 ou 2 minutos em imagens grandes.

## Decisões de implementação

- A conversão RGB para YCrCb é feita com a matriz da equação (1), usando só o canal Y.
- A imagem é convertida para `float64` antes de subtrair 128, para evitar overflow de `uint8`.
- O padding usa a replicação das bordas e é removido no final da reconstrução.
- O bpp usa as dimensões originais da imagem, sem contar o padding.
- A porcentagem de coeficientes zero considera todos os coeficientes quantizados.
- A DCT rápida usa `norm="ortho"` para ter a mesma normalização da matricial.
- O PSNR é calculado diretamente. O SSIM usa a função do `skimage` com `data_range=255`.

## Observações

- Em alguns valores de `s`, as imagens das versões A e B podem diferir muito pouco no PSNR e no SSIM. As duas DCTs diferem em torno de 1e-13, o que pode mudar o arredondamento de alguns coeficientes na quantização.
- O speedup medido pode ser menor que 1. Isso acontece porque `C @ bloco @ C.T` com blocos de 8x8 já roda em código compilado do NumPy, e a chamada da função rápida tem overhead. Esse ponto faz parte da discussão pedida no enunciado.
