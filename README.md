# Imersão dados 4
Neste conjunto de aulas promovida pela Alura em parceria com a Creditas, ocorridas entre os dias 23/05/22 e 27/05/22, tivemos a oportunidade de explorar uma base de dados de imóveis na cidade de São Paulo. Dentre as tantas tarefas, fizemos uma exploração preliminar dos dados, limpeza da base, adicionamos dados externos e relacionamos os dados com regressores e machine learning para atingir o objetivo principal: desenvolver um modelo capaz de estimar, a partir de um conjunto de informações, o preço de um imóvel. Com intuito de expor o que foi praticado com Python, trago um resumo do .ipynb utilizado para estudo e presente neste repositório.

![image](https://user-images.githubusercontent.com/99848777/171287224-9762fdf2-abd4-4584-9ef4-17c729a0670e.png)

# O primeiro contato com os dados
Começamos acessando o [gist](https://gist.githubusercontent.com/tgcsantos/3bdb29eba6ce391e90df2b72205ba891/raw/22fa920e80c9fa209a9fccc8b52d74cc95d1599b/dados_imoveis.csv) cuja visualização inicial nos apresenta o seguinte *head*:

![image](https://user-images.githubusercontent.com/99848777/171492516-4519f01c-23e6-48fc-aabf-ab761809e04e.png)

As primeiras interações com o *dataframe* envolveram conhecer funcionalidades do Pandas, e o primeiro problema identificado é a impossibilidade de manipular os numeros refentes ao valor do imóvel, que estão classificados como objeto por conta do "R$". O problema pode ser resolvido pelo ``.strip()`` ou, preferencialmente, pelo ``.split(expand=True)``, pois nossos dados possuem marcadores de aluguel que não são desejados, e o `.split()` ajuda a filtrar ao criar uma coluna para os marcadores de aluguel. Enfim, depois desse primeiro tratamento, é possível ver a distribuição dos preços dos imóveis com auxílio do `sns.histplot()`.

![image](https://user-images.githubusercontent.com/99848777/171509418-9bdd6d2f-096b-4376-a7a0-a65fc12b96d9.png)

Observa-se que a grande maioria dos imóveis do *dataframe* se encontra na faixa de até R$ 5 milhões, mas ainda temos registro de imóveis com valor acima disso. Inclusive, a base contém muitos *outliers*, como é observável a partir dos *boxplots* da distribuição de cada um dos dados numéricos, feito com auxílio do `sns.boxplot()`.

![image](https://user-images.githubusercontent.com/99848777/171511074-95c3beee-04a7-4ec3-a30d-ffcdda04403b.png)

Existe até um *outlier* na Metragem que inviabiliza a apresentação do *boxplot*. São *outliers* também identificáveis quando cruzamos alguns dados, como a distribuição do preço do metro quadrado em relação à quantidade de banheiros do imóveil:

![image](https://user-images.githubusercontent.com/99848777/171511689-9987f163-14aa-4e3f-a900-56fb15774fdb.png)

# Adicionando dados relevantes para a análise
Um tipo de dado que poderia ajudar compreender os preços dos imóveis é de características econômicas do local. O curso ofereceu o presente [gist](https://gist.githubusercontent.com/tgcsantos/85f8c7b0a2edbc3e27fcad619b37d886/raw/a4954781e6bca9cb804062a3eea0b3b84679daf4/Basico_SP1.csv) de dados do Censo 2010 do IBGE, com o presente [dicionario](https://drive.google.com/file/d/1WVTqfKtHOOk5X1AWaSOn6NLaO7cix2m4/view). O desafio é conseguir relacionar os endereços com os setores censitários para ter as informações econômicas do local.
Uma primeira opção, menos trabalhosa, porém menos precisa, é relacionar o nome de bairros e distritos. O problema é que a base apresenta uma quantidade de nome de bairros muito distinta da relação oficial de distritos no município de São Paulo. Então muitas linhas se perdem ao tentar fazer essa relação, e perdemos na variabilidade de dados de renda (que são os dados observados do Censo aqui).

![image](https://user-images.githubusercontent.com/99848777/171516049-133ab197-7d9e-4c75-a0d8-7f31e7deb51e.png)

Podemos ver muitos dados enfileirados, indicando a falta de variabilidade. Por exemplo, todo imóvel localizado na Vila Mariana indica a mesma renda média neste gráfico, só podemos ver a variabilidade do preço do metro quadrado pra essa dada renda. Para poder superar a simplicidade da relação a partir dos bairros, o curso disponibilizou um *CSV* com endereços do município, com localização de latitude e longitude, que é muito grande para compartilhar aqui. O processo envolveu os nomes das ruas e seus devidos CEPs, de modo que nosso *dataframe* original pudesse conter também dados de geolocalização.
Com apoio do pacote Geopandas e arquivos de mapas também disponibilizados pelo curso (e igualmente muito grandes para compartilhar por aqui), foi possível localizar os pontos gerados pelo cruzamento de latitude e longitude e obter os setores censitários de cada imóvel da base. Nosso *dataframe* agora possui colunas essenciais para adicionar integralmente os dados do IBGE citados anteriormente.

# Dataframe com os dados por setor censitário

O gráfico que apresentamos anteriormente agora se transforma neste gráfico, onde a renda varia mais livremente:

![image](https://user-images.githubusercontent.com/99848777/171519088-c35d101b-1ff3-45e0-8bb0-baecfa908296.png)

Agora temos uma grande quantidade de dados disponíveis no *dataframe*, e isso permite vislumbrar relacionamentos entre os dados. Para tanto, produzimos uma matriz de correlações para identificar quais dados tem um relacionamento potencial.

![image](https://user-images.githubusercontent.com/99848777/171522658-204ce608-fb5a-4cb9-8821-61e0f0cc93b0.png)

Existem algumas correlações altas e pouco significativas, como as diferentes formas do IBGE de calcular rendimentos mensais dos setores censitários, e do preço do metro quadrado com o preço do imóvel. Outras, como quantidade de banheiros com quantidade de quartos, que não são tão altas mas demonstram que existe algum padrão distributivo de cômodos (exemplo: uma casa com cinco quartos deve ter mais banheiros do que uma casa com apenas um quarto). Alguns dados apresentam mais interesse, como o rendimento mensal do método V009 e algumas características da casa como quartos, banheiros, preço do imóvel, além do preço do metro quadrado - ao mesmo tempo que o rendimento mensal não demonstra nenhuma relação com a metragem dos imóveis.

# Modelos com scikit-learn

Um primeiro modelo proposto pelo curso foi apenas para apresentar o pacote *scikit-learn*, e relacionava linearmente o preço dos imóveis como a variável explicada e a metragem como a variável explicativa. De cara, apresentou um erro médio absoluto muito elevado, então foram incluídas mais variáveis explicativas: quartos, banheiros, vagas de garagem, e rendimento médio mensal nas metodologias V001, V007 e V009. O erro médio absoluto e o R² comparando a previsão com os valores reservados para teste foram:
>Erro médio absoluto do teste: 1355585.1902992018

>R² do teste: 0.37760744704151483

Apesar de ainda ser um modelo modesto, ele aparentemente captou as tendências, como visto no gráfico de comparação dos valores reais e previstos:

![image](https://user-images.githubusercontent.com/99848777/171526394-7aa5ed57-a077-4793-9c12-d15c237e68bc.png)

O modelo teve dificuldade de prever valores mais altos. Optei por testar a regressão utilizando Random Forest Regressor, que, segundo a documentação do *scikit-learn*:
>A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

Os valores de teste já foram consideravelmente maiores:

>Erro médio absoluto do teste: 712571.7069270463

>R² do teste: 0.7614556859788321

Mas, buscando refinar ainda mais o modelo, foi utilizado o GridSearchCV junto com Pipeline para definir os melhores parâmetros, mas o ganho de refino não foi substancial, os valores de teste ficaram:

>Erro médio absoluto do teste: 703516.794781712

>R² do teste: 0.7745624078977799

Este último modelo já foi mais sagaz na predição de valores altos, como se pode ver:

![image](https://user-images.githubusercontent.com/99848777/171527124-d317e05c-9f31-4f5b-bd65-5cd2483f406d.png)

E se constatou, através do *feature importances*, quais foram as variáveis explicativas mais significativas neste modelo:

![image](https://user-images.githubusercontent.com/99848777/171527236-b585d024-b9a3-44c3-aaec-d11e83ef5cca.png)



Com isso, temos que a metragem se destaca na explicação do valor do imóvel, seguida das medidas de renda da região do imóvel.

Enfim, essa modelagem nos dá uma boa chance de adivinhar o valor do imóvel com algumas informações. O gráfico onde cruzamos o valor real dos imóveis com o valor que o modelo previu tem grande parte da linha alaranjada sobreposta à linha azul. Quando comparado com o gráfico referente ao modelo de regressão linear, vemos um avanço substancial de previsibilidade.

# Considerações finais

Este projeto resume brevemente a semana de trabalhos do *datacamp*. Os desafios trouxeram grandes aprendizados e, a despeito do modelo ter sido feito em pouco tempo, e possivelmente termos falhas tanto na construção do modelo como na preparação dos dados, o resultado final foi muito responsivo. Os pacotes de análise de dados para Python que utilizamos em conjunto foram: *numpy, pandas, matplotlib, seaborn e geopandas*. A plataforma do Jupyter Notebook foi o *Google Colaboratory*. Todo processo está detalhado no .ipynb deste repositório.
