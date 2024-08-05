# tp2_ml_adaboost 

Replicando e discutindo o algoritmo Adaboost no dataset Tic-Tac-Toe (ucimlrepo).

### Especificação de objetivo:
Solucionar o problema de classificação binária no dataset tic-tac-toe, incluindo validação cruzada com 5 partições, com erro simples como medida de eficácia. <br>
O modelo deve ser construído através de boosting de stumps (árvores de decisão com uma só "pergunta").

### Especificação de entrega:
Código fonte e documentação do programa que implementa boosting e solução do dataset, demonstrando gráficos de erro de treino e teste do modelo e dos stumps.

### Resultados:

O algoritmo apresentou o comportamento esperado a partir das discussões em sala de aula, de forma que o erro de cada stump aumenta (tendendo a 0.5) ao passo que o erro empírico e de validação reduzem, com erro empírico menor ou igual ao de validação.

Conforme esperado - em validação cruzada - é possível observar alguma variabilidade nos resultados entre um ensemble e outro, principalmente no erro de validação. Porém, a tendência do comportamento é a mesma: o erro do stump se aproxima de 50% à medida que o erro empírico é minimizado (e por consequência, também o de validação) de tal forma que a decisão de stumps de iterações finais tem cada vez menor importância mas a decisão coletiva melhora (erro de validação diminui).

Além disso, mesmo executando um número elevado de iterações (maior que o número de exemplos no banco de dados), o erro de validação não desviou do erro empírico, ilustrando a resistência por design do algoritmo Adaboost a overfitting. Vemos também que os pesos aumentam gradativamente, priorizando aprendizado de instâncias mais difíceis (onde o modelo ainda erra), e que a importância (alpha) de cada decisor cai de maneira correspondente, conforme esperado.

Em resumo, é demonstrada a capacidade do algoritmo Adaboost de produzir um conjunto de "classificadores fracos" que agregados por ponderação são capazes de otimizar a eficácia do modelo sem causar overfitting (em teoria).

## Ambiente:
1. usando conda, criado um ambiente python 3.11
2. instalado poetry no ambiente conda
3. instalado pandas, numpy, matplotlib, seaborn, pytorch, optuna, mlflow, jupyter, ipykernel

### Como:

Abra um terminal na pasta onde vai criar o notebook e execute:

``` bash 
conda create -n tp1ml python=3.11 -y
```

``` bash 
conda activate tp1ml
```

``` bash 
pip install poetry
```

``` bash 
poetry new tp1ml_mnist_proj
```


``` bash 
cd tp1ml_mnist_proj
```

``` bash 
poetry config virtualenvs.create false --local
```

``` bash 
poetry add pandas numpy matplotlib seaborn torch optuna mlflow jupyter ipykernel tensorboard
```

Opcional:

``` bash 
poetry show
``` 

### Troubleshoot: 
if installation fails on Windows, try running on cmd instead of PowerShell (windows terminal). Not sure why yet, but  that doesn't work.
