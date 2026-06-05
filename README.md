# GS 2026/1 — Sistemas Operacionais — Indústria Aero-espacial

Projeto da Global Solution 2026/1 da disciplina de **Sistemas Operacionais**
(Prof. Dr. José Gomes Salim Neto — FIAP).

Quantificamos a influência dos padrões numéricos `int8` e `float64` em
processamentos iterativos embarcados (rede neural MLP) executados em um
sistema operacional pré-emptivo com Kernel de 64 bits, no contexto da
estimativa de **distância de parada na corrida de pouso** de um jato comercial.

## Integrantes

Nome e RM
Miguel Leal 553009 
João Víctor Flaitt  553888 
Lucas Bertolassi  553183 
Lucca Calsolari  553678 

## Estrutura

```
.
├── mlp_pouso.py        # pipeline completo: dataset, MLP, treino, plots
├── history.json        # histórico de métricas (gerado pelo run)
├── figs/               # gráficos (gerados pelo run)
└── README.md
```

## Como executar

```bash
pip install numpy matplotlib
python mlp_pouso.py
```

O script:

1. Gera o **dataset sintético** com 6 000 000 de linhas e 5 atributos
   (V0, massa, μ da pista, gradiente, altitude do aeródromo) usando
   um modelo físico (RK4 sobre EDO não linear) + extrapolação polinomial
   ajustada por OLS.
2. Faz **exploração, tratamento e preparação** (winsorize 1 %, split 70/15/15,
   normalização Min-Max).
3. Treina uma **MLP simples** (1 camada oculta de 16 neurônios + 1 saída,
   sigmoide em ambas) duas vezes — em `float64` e em `int8` — com 200
   épocas, batch 4096, LR 0.05.
4. Salva curvas de **MSE / R²** por época e o **tempo total em minutos**
   em `figs/curvas_treinamento.png` e `figs/tempo_total.png`.

## Métricas

* MSE de treino e validação;
* MAE de validação;
* R² de validação;
* Tempo total de processamento (min).

## Equações cinemáticas

A dinâmica da corrida de pouso é modelada por

$$ m\,\frac{dV}{dt} = -\tfrac12 \rho V^{2} S\,C_d - \mu\,(m g\cos\theta - \tfrac12 \rho V^{2} S\,C_l) - m g \sin\theta $$

ou seja, **movimento variado não uniformemente** (a aceleração depende
explicitamente de V), conforme exigido no enunciado. A integração é feita
por Runge-Kutta 4 com passo Δt = 0,05 s.