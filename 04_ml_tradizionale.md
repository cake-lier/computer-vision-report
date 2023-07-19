# Classificazione tradizionale

## Preparazione dei dataset

Una volta determinate le *feature* da estrarre, non si è fatto altro che effettuare la loro elaborazione per ogni immagine a colori corretta, indipendentemente dal fatto che quest'ultima appartenesse al *training set* o al *test set*. Nel caso in cui da un'immagine non fosse possibile estrarre una o più *feature*, questa viene interamente scartata.

Una volta estratte le *feature*, è stato necessario normalizzarle. I descrittori che sono realizzati come istogrammi non possono però esserlo, in quanto ciascuna componente ha un significato che è legato a quello delle altre. Riscalare i valori presi singolarmente altererebbe i rapporti tra questi, facendo perdere parte del significato che portano con sé. Tanto più che la normalizzazione degli istogrammi viene già fatta, semplicemente dividendo il valore di ciascun "bin" per la somma dei valori, dando origine ad una funzione di distribuzione di probabilità discreta.

Discorso analogo vale per i descrittori di Fourier ellittici: sono coefficienti che codificano informazioni sullo "spettro delle frequenze" di una forma in termini di ellissi la cui somma restituisce la silhouette originale. Non sono perciò importanti solamente i rapporti tra questi, ma anche il loro valore assoluto, che indica con che ampiezza una data frequenza è presente nel segnale di partenza. Si ricorda inoltre che la loro estrazione già prevede una fase di normalizzazione degli stessi.

I descrittori che non ricadono in queste categorie sono invece i centroidi, che sono stati resi indipendenti dalle dimensioni dell'immagine da cui sono stati estratti, ma non sono normalizzati. Analogamente non sono normalizzati l'elongazione e la convessità, così come i descrittori del tono della pelle del volto, che sono distanze, quindi possono variare tra $0$ e $+\infty$.

Per questo motivo, tutti questi descrittori sono stati normalizzati mediante "standard scaling", ovvero tramite la sottrazione della media dei valori e divisione per la deviazione standard. In questo modo, la distribuzione dei valori di ciascun descrittore, se può essere espressa mediante una funzione gaussiana, è una gaussiana di media 0 e varianza 1.

## Addestramento del classificatore per il primo problema

Una volta normalizzate le *feature*, si è trattato di sostituire le *feature* delle immagini alle stesse nel *dataset* di *training* delle coppie del primo problema. Viene quindi costruito un nuovo *dataset* che ha tante righe quante sono le istanze dalle quali è stato possibile estrarre tutte le *feature* e un numero di colonne pari al doppio delle *feature* per ciascuna immagine. Questo perché le *feature* estratte da ciascuna delle due immagini della coppia vengono affiancate in questo nuovo *dataset*. Nel *dataset* originale si trattengono solamente le coppie di immagini per cui le *feature* sono state estratte da entrambe, così da avere le etichette delle classi. La stessa cosa viene fatta anche per il *dataset* di *test* delle coppie per il primo problema.

Le operazioni di estrazione variano leggermente le percentuali sul totale delle due classi, sia per il *training set* che per il *test set*. Ciò ci permette di non effettuare ulteriori operazioni di aggiustamento delle istanze, ma verranno considerati i diversi pesi di ciascuna classe durante l'addestramento.

Table: Numero di istanze per le classi negativa e positiva nel _training set_ dopo l'estrazione delle _feature_

|label|count|
|---|---|
|0|    44396|
|1|    41637|

Table: Numero di istanze per le classi negativa e positiva nel _test set_ dopo l'estrazione delle _feature_

|label|count|
|---|---|
|0|    6039|
|1|    6255|

Le funzioni che possono essere utilizzate per combinare le *feature* delle coppie di immagini sono:

* somma: $x + y$
* prodotto: $x \cdot y$
* differenza: $x-y$
* somma al quadrato: $(x+y)^2$
* differenza al quadrato: $(x-y)^2$
* somma dei quadrati: $x^2 + y^2$
* differenza dei quadrati: $x^2 - y^2$

Queste funzioni vogliono essere delle versioni semplificate di metriche di distanza o di similarità. In generale, somme o prodotti "_element-wise_" restituiscono valori tanto più elevati quanto più gli operandi sono vicini tra loro, le differenze valori tanto più grandi quanto più gli operandi sono diversi tra loro. Si è voluto evitare l'utilizzo dell'operazione di divisione, che non è definita nel caso di divisione per 0, così come la moltiplicazione o la somma di costanti, che avrebbe solamente modificato lo *scaling* delle *features*. L'uso di queste funzioni è stato ispirato dal vincitore della competizione, che utilizzava alcune di esse per combinare le *feature* estratte dalle diverse reti neurali. Infatti, questo ha permesso lui di ottenere modelli sufficientemente indipendenti tra di loro, per poterli utilizzare come "*ensemble*".

Una volta definiti i *dataset* da utilizzare e le funzioni, è stato effettuato l'addestramento vero e proprio per ottenere i migliori modelli. La famiglia di classificatori utilizzati è "XGBoost", in quanto capace di poter addestrare concorrentemente i *decision tree* che fanno parte degli "*ensemble*" che costruiscono sulla GPU. Questo rende possibile effettuare efficientemente il loro addestramento su *cluster* dotati di molte risorse di GPU.

È necessario specificare la funzione obiettivo, che altro non è che una regressione logistica che permette di effettuare classificazione. Il problema è a due sole classi, perciò è una "logistic regression" binaria. La metrica scelta per valutare l'errore del modello sul *validation set* è il più semplice tasso di errore, ovvero $\frac{FP + FN}{P + N}$.

È stata effettuata una "*grid search*" per ottenere i migliori iperparametri in corrispondenza dell'applicazione di ciascuna delle funzioni per l'aggregazione delle *feature* corrispondenti. I valori che sono stati lasciati qui indicati sono stati quelli che si sono rivelati migliori dopo diversi tentativi su differenti intervalli. Nella *grid search* gli iperparametri che si sono rivelati più significativi sono stati:

* "min_child_weight": il peso minimo che deve essere associato ad una foglia di un albero costruito, valori più grandi portano ad una terminazione maggiormente precoce del processo di costruzione-suddivisione di ogni singolo albero;
* "gamma": minima riduzione della funzione di *loss* definita perché un'ulteriore suddivisione di una foglia possa essere considerata;
* "subsample": percentuale del *training set* da campionare per la costruzione degli alberi, fatto per ridurre il potenziale *overfitting*;
* "colsample_bytree": la percentuale di colonne del *training set* da campionare durante la costruzione degli alberi, per ogni nuovo albero costruito;
* "max_depth": la profondità massima di un singolo albero, regola la complessità dello stesso e con essa la possibilità di *overfitting*;
* "learning_rate": parametro moltiplicativo che ha per scopo ridurre mano a mano il peso dei risultati degli alberi costruiti, in modo tale da evitare "salti" troppo grandi nella ricerca dei parametri ottimali del modello;
* "n_estimators": il numero massimo di alberi che verranno addestrati da XGBoost.

Il vero e proprio "*estimator*" addestrato durante la *grid search* è stato costruito mediante una "Pipeline" che applica in sequenza la funzione di aggregazione dei dati scelta e in seguito il modello da addestrare. Questo implica quindi che viene costruita una "Pipeline" per ognuna delle possibili funzioni.

La tecnica di *grid search* impiegata è la cosiddetta "Bayesian Optimization", ovvero una tecnica iterativa che si basa sulla formula di Bayes per trovare i massimi di una funzione sconosciuta a partire da un insieme di campionamenti limitato perché "costosi" da calcolare.
L'obiettivo è quello di individuare un insieme di valori, un insieme di input, su un dominio limitato da usare come "sonda". Occorre tenere conto sia di dove si ha maggiore incertezza sulla forma, sull'output, della funzione, sia di dove ci si aspetta che siano presenti i massimi della funzione, usando le informazioni già accumulate ed euristiche adeguate. Ecco quindi perché è coinvolto il teorema di Bayes, che infatti permette di calcolare la probabilità condizionata di un evento rispetto ad un secondo già accaduto, a priori, conoscendo la probabilità che quello già accaduto si verifichi.

Nel nostro caso, la funzione di cui si sta cercando di individuare il massimo è il modello che si sta addestrando che prende per input vettori nello spazio degli iperparametri e come output restituisce l'accuratezza sul *training set*, che deve essere massima. La tecnica non farà altro che campionare il modello in corrispondenza dei punti per lui sensati alla ricerca del massimo.

Si è osservato come l'uso di questa tecnica, in corrispondenza dei parametri scelti, abbia accelerato l'effettuazione della *grid search*. È stato infatti specificato di tentare 100 vettori nello spazio degli iperparametri, testandone 4 concorrentemente alla volta. La valutazione delle prestazioni degli iperparametri è stata effettuata considerando l'accuratezza come *score* e facendo "stratified k-fold cross validation". Questo significa che per ogni combinazione degli iperparametri il *training set* è stato suddiviso in tre parti: due sono state utilizzate per l'addestramento vero e proprio e una per la valutazione, cambiandole ogni volta in modo tale da poter ottenere alla fine tre *score* su cui trarre un giudizio. Il fatto che la *cross validation* fosse "stratified" ha permesso di mantenere la stessa proporzione tra le due classi in ciascuno dei *fold*, cioè delle partizioni.

I risultati dell'addestramento dei modelli che fanno uso della funzione somma ci dicono che i migliori 5 modelli hanno uno *score* medio sugli *split* che non si discosta molto dal 64,5%. L'addestramento ha prediletto un *ensemble* molto numeroso, con un numero di alberi nell'ordine del migliaio, relativamente profondi, tutti con una profondità massima pari al valore più alto scelto. Il *learning rate* scelto è stato molto basso, segno che è stato preferito un approccio conservativo. L'addestramento ha preferito non scartare una percentuale di colonne e di istanze del *dataset* di *training* molto elevata in corrispondenza di ogni nuovo albero, segno che non è stato individuato particolare *overfitting*. Da ultimo, anche il valore del parametro "gamma" rimane particolarmente ridotto tra i modelli migliori.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la somma come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
| 0.8021    | 0\.0268 | 0\.01    | 15      | 4          | 846           | 1\.0      | 0\.6475    |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| 1\.0      | 2\.4435 | 0\.0180  | 13      | 5          | 1000          | 0\.8662   | 0\.6461    |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| 0\.2332   | 0\.01   | 0\.01    | 12      | 10         | 852           | 0\.6913   | 0\.6458    |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| 1\.0      | 0\.01   | 0\.01    | 15      | 10         | 1000          | 1\.0      | 0\.6458    |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| 1\.0      | 0\.01   | 0\.0114  | 15      | 6          | 758           | 0\.7849   | 0\.6456    |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

I risultati dell'addestramento dei modelli che fanno uso della funzione prodotto ci dicono che i migliori 5 modelli hanno uno *score* medio sugli *split* che non si discosta molto dal 64,5%. I risultati sono simili a quelli precedenti, con le principali differenze legate al fatto che l'*ensemble* in questo caso è mediamente ancora più numeroso e con alberi più profondi. Il numero di colonne e di istanze del *dataset* considerate per ogni albero si mantiene alto, come è alto, ma solo in questo caso, il peso che ogni foglia dell'albero deve avere. Il parametro *gamma* mantiene dei valori relativamente ridotti come in precedenza, mentre il *learning rate* ci dice che l'addestramento mediamente è stato meno conservativo.

\newpage

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano il prodotto come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.6653    |0\.0528  |0\.0235   |14       |10          |729            |0\.7689    |0\.6468     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0237   |15       |10          |1000           |1\.0       |0\.6467     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.0356   |15       |10          |1000           |1\.0       |0\.6462     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.6058    |0\.01    |0\.0127   |15       |1           |1000           |0\.8643    |0\.6459     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.01     |15       |10          |1000           |0\.5184    |0\.6459     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

I risultati dell'addestramento dei modelli che fanno uso della funzione di differenza ci dicono che i migliori 5 modelli hanno uno *score* medio sugli *split* che non si discosta molto dal 66%. L'addestramento ha sempre prediletto un *ensemble* molto numeroso con alberi relativamente profondi. Il *learning rate* scelto è stato molto basso, segno che è stato preferito un approccio conservativo. L'addestramento ha preferito non scartare una percentuale di istanze del *dataset* di *training* molto elevata in corrispondenza di ogni nuovo albero, ma ha voluto trattenere molte poche colonne. Da ultimo, il valore del parametro "gamma" ha un atteggiamento un po' ondivago, anche se rimane contenuto, così come il peso minimo delle foglie dell'albero.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la differenza come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.06      |0\.01    |0\.0167   |13       |1           |1000           |1\.0       |0\.6591     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.0206   |15       |1           |1000           |0\.7960    |0\.6586     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |1\.0750  |0\.0120   |12       |10          |1000           |1\.0       |0\.6581     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.1149  |0\.0242   |15       |9           |1000           |1\.0       |0\.6580     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |3\.9005  |0\.0239   |12       |1           |728            |0\.7261    |0\.6575     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

I risultati dell'addestramento dei modelli che fanno uso della funzione somma al quadrato ci dicono che i migliori 5 modelli hanno uno *score* medio sugli *split* che non si discosta molto dal 64%. L'addestramento ha prediletto come sempre un *ensemble* molto numeroso con alberi relativamente profondi. Il *learning rate* scelto è stato molto basso, segno che è stato preferito un approccio conservativo, così come il parametro "gamma", che nei migliori modelli è sempre stato fisso sul minimo possibile. L'addestramento ha preferito non scartare una percentuale di colonne e di istanze del *dataset* di *training* molto elevata in corrispondenza di ogni nuovo albero, segno che non è stato individuato particolare *overfitting*, anche se mediamente rispetto ai casi predenti il numero di istanze considerate per ciascun albero è stato più contenuto, sempre al di sotto dell 86%. Anche qui notiamo la mancanza di una tendenza precisa per quanto riguarda il peso minimo delle foglie.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la somma al quadrato come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|1\.0       |0\.01    |0\.0220   |15       |10          |729            |0\.7425    |0\.6429     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0232   |11       |1           |1000           |0\.7678    |0\.6429     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0176   |15       |1           |1000           |0\.8580    |0\.6425     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0136   |15       |10          |725            |0\.5134    |0\.6425     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.4616    |0\.01    |0\.01     |15       |5           |1000           |0\.5998    |0\.6422     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

I risultati dell'addestramento dei modelli che fanno uso della funzione differenza al quadrato ci dicono che i migliori 5 modelli hanno uno *score* medio sugli *split* che non si discosta molto dal 64%. Questo è il caso in cui l'addestramento ha prediletto un *ensemble* più numeroso possibile e tutti gli alberi quanto più profondi possibile. Il *learning rate* scelto è rimasto stabilmente attorno al minimo possibile. L'addestramento ha preferito non scartare colonne del *dataset* di *training* in corrispondenza di ogni nuovo albero, ma di campionare almeno in parte le sue istanze, per controllare un potenziale *overfitting*. Da ultimo, anche il valore del parametro "gamma" rimane particolarmente ridotto tra i modelli migliori.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la differenza al quadrato come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|1\.0       |0\.1631  |0\.01     |15       |1           |1000           |0\.8290    |0\.6430     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0169   |15       |10          |1000           |0\.7590    |0\.6428     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.0654  |0\.0154   |15       |1           |1000           |0\.7125    |0\.6420     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.0242   |15       |10          |1000           |0\.8487    |0\.6420     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.01     |13       |1           |1000           |0\.6564    |0\.6417     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

I risultati dell'addestramento dei modelli che fanno uso della funzione somma di quadrati ci dicono che i migliori 5 modelli hanno uno *score* medio sugli *split* che non si discosta molto dal 64%. Come in tutti i casi precedenti, l'addestramento ha prediletto un *ensemble* molto numeroso con alberi relativamente profondi. L'approccio è stato molto conservativo, con il *learning rate* fisso sul minimo possibile, così come il parametro "gamma". Questa volta, il comportamento indeciso si riscontra sia sulla percentuale di colonne da utilizzare nella costruzione di un nuovo albero, sia sul peso minimo di una foglia, che in parte anche sulla percentuale di istanze da campionare, anche se rimane sempre alto e sopra il 56%.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la somma di quadrati come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.06      |0\.01    |0\.01     |12       |1           |1000           |1\.0       |0\.6433     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.01     |13       |10          |1000           |1\.0       |0\.6430     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.01     |14       |10          |1000           |0\.5909    |0\.6422     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.6747    |0\.01    |0\.01     |15       |7           |804            |0\.5596    |0\.6419     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.01     |13       |1           |1000           |0\.7814    |0\.6419     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Da ultimo, si è considerato l'addestramento che ha coinvolto i modelli che hanno fatto uso della funzione differenza di quadrati. In questo caso, l'addestramento sembra essere stato molto deciso: numero di alberi massimo per ogni *ensemble*, profondità massima degli stessi sempre vicina al massimo, *learning rate*, parametro "gamma" e percentuale di colonne per albero minimi. La percentuale di istanze campionate per albero rimane elevata, anche se non sempre massima, segno di un basso *overfitting*, mentre come spesso è successo il peso minimo di una foglia varia considerevolmente tra i migliori modelli.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la differenza di quadrati come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.06      |0\.01    |0\.01     |15       |10          |1000           |1\.0       |0\.6555     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.01     |13       |1           |1000           |0\.7035    |0\.6543     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.01     |12       |10          |1000           |1\.0       |0\.6538     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.01     |14       |10          |1000           |0\.6889    |0\.6536     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.0153   |11       |1           |1000           |0\.6978    |0\.6532     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Una volta identificato il migliore modello in corrispondenza di ciascuna funzione, ognuno di questi viene riaddestrato sull'intero *dataset* di *training* per poter valutare le sue prestazioni sul *dataset* di test. Gli *score* per ciascun modello mostrano un discreto *overfitting*, con delle differenze tra accuratezza di *training* e di *test* che oscillano attorno a 30-40 punti percentuali. La differenza minore si ha per le funzioni che impiegano la differenza, con uno scarto tra le accuratezze mai superiore al 25%. In generale comunque, i risultati sono abbastanza scoraggianti: le accuratezze sul *test set* non superano mai il 55%, segno che i modelli non sono stati in grado di generalizzare e di apprendere qualcosa di significativo dalle *feature*. Questo lo si nota anche dalle "*confusion matrix*", dove osservando i positivi e i negativi predetti, sono sempre all'incirca lo stesso numero, sia che lo fossero davvero sia che non lo fossero, con una predilezione generale per la classe dei negativi. Osservando le matrici, il modello che tendenzialmente sbaglia di meno è di nuovo quello che si basa sulla differenza, seguito dalla differenza al quadrato e dal quadrato delle differenze.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/sum_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 90\%, quella di 
    \textit{test} 53\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/prod_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione prodotto. L'accuratezza di \textit{training} era 88\%, quella di 
    \textit{test} 54\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/diff_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione differenza. L'accuratezza di \textit{training} era 69\%, quella di 
    \textit{test} 54\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/squared_sum_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma al quadrato. L'accuratezza di \textit{training} era 89\%, quella di 
    \textit{test} 53\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/squared_diff_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione differenza al quadrato. L'accuratezza di \textit{training} era 79\%, quella di \textit{test} 55\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/sum_squares_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma di quadrati. L'accuratezza di \textit{training} era 89\%, quella di 
    \textit{test} 52\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/diff_squares_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione differenza di quadrati. L'accuratezza di \textit{training} era 71\%, quella di \textit{test} 53\%}
\end{figure}
```

Anche in questo caso, per velocizzare il processo di addestramento, è stato utilizzato come "classificatore finale" un modello della famiglia XGBoost, lo stesso che in precedenza. Gli iperparametri da tarare per esso sono quindi gli stessi che in precedenza. Se i classificatori che forniscono il loro output sono stati determinati al passo precedente, quello finale che compone i loro risultati viene determinato tramite una seconda *grid search* con "Bayesian Optimization", che riutilizza gli stessi valori per gli iperparametri della precedente.

I risultati dell'addestramento mostrano come il modello finale risulti troppo complesso e affetto da chiaro *overfitting*, in quanto lo *score* medio tra tutti i modelli non sia mai inferiore al 99,99%. Questo si nota anche dai valori degli iperparametri per quelli che teoricamente sono i migliori modelli: in praticamente nessuno di essi si nota una chiara tendenza. La percentuale di istanze campionate, il numero di alberi, il peso minimo di una foglia, la profondità massima di un albero e la percentuale di colonne considerate per un nuovo albero possono essere alti o bassi indifferentemente rispetto allo *score* finale. Gli iperparametri che mostrano meno variabilità sono il *learning rate*, che non supera mai il 45% e il parametro "gamma".

Table: Risultati dell'addestramento dei migliori cinque modelli finali, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.4468    |4\.6909  |0\.0235   |13       |7           |968            |0\.4286    |0\.9999     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.8968    |0\.0499  |0\.3585   |4        |3           |221            |0\.9168    |0\.9999     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.1335    |0\.0347  |0\.4341   |9        |4           |834            |0\.2299    |0\.9999     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.6436    |0\.01928 |0\.1110   |5        |8           |740            |0\.9540    |0\.9999     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.3954    |0\.0832  |0\.2860   |7        |5           |124            |0\.1271    |0\.9999     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Scelto quello che teoricamente è il miglior modello addestrato, viene poi valutato sul *test set*. Il risultato finale mostra quello che già abbiamo notato in precedenza: l'accuratezza sul *training set* è oltre il 90%, ma quella sul *test set* è attorno al 50%, segno che non ha generalizzato nulla ed ha imparato i *pattern* "a memoria". Anche in questo caso, la rete preferisce catalogare tutte le istanze come negativi, avendo una percentuale di veri positivi e veri negativi molto bassa.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/final_bin_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello finale. L'accuratezza di \textit{training} era 93\%, quella di \textit{test} 53\%}
\end{figure}
```

Vengono qui di seguito mostrati i cinque casi in cui il modello ha dato i migliori risultati in ordine decrescente, cioè prima i migliori in assoluto. Come si può osservare il modello classifica tutte le coppie come "non parenti". Questo è in linea con la *confusion matrix* mostrata in precedenza e indica che il modello ha appreso poco dalle *feature* e classifica la maggior parte delle coppie di persone come "non parenti". Tra le *feature* che noi riteniamo che il modello utilizzi maggiormente per classificare siano il colore della pelle, la posa e/o posizione del volto e la forma degli occhi. La maggiore risoluzione delle immagini permette una estrazione migliore delle *feature* e un conseguente miglioramento nelle prestazioni del modello.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{images/top_bin_1.png}
    \includegraphics[width=0.45\textwidth]{images/top_bin_2.png}
    \includegraphics[width=0.45\textwidth]{images/top_bin_3.png}
    \includegraphics[width=0.45\textwidth]{images/top_bin_4.png}
    \includegraphics[width=0.45\textwidth]{images/top_bin_5.png}
    \caption{Le cinque istanze in cui il modello finale ha dato i migliori risultati}
\end{figure}
```

Se osserviamo le istanze per le quali il modello predice la classe "parente" commettendo minore errore pensiamo si possa osservare che, oltre alle caratteristiche individuate nel punto precedente, anche la forma della bocca e un sorriso che mostra i denti contribuiscano alla classificazione corretta.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{images/top_rel_bin_1.png}
    \includegraphics[width=0.45\textwidth]{images/top_rel_bin_2.png}
    \includegraphics[width=0.45\textwidth]{images/top_rel_bin_3.png}
    \includegraphics[width=0.45\textwidth]{images/top_rel_bin_4.png}
    \includegraphics[width=0.45\textwidth]{images/top_rel_bin_5.png}
    \caption{Le cinque istanze della classe dei positivi in cui il modello finale ha dato i migliori risultati}
\end{figure}
```

Vengono qui di seguito mostrati i casi in cui il modello ha dato i peggiori risultati in ordine decrescente, prima i peggiori in assoluto.
Specularmente rispetto a prima, gli errori maggiori commessi dal modello riguardano in maggioranza le coppie legate da una parentela. Tra le caratteristiche delle immagini che portano il modello a compiere errori c'è il fatto che sono rimaste immagini "_grayscale_" dal processo di *preprocessing* delle immagini. Altri fattori che pensiamo abbiano inciso negativamente sulla qualità della predizione sono la bassa qualità dell'immagine, la differente illuminazione tra le due immagini, la presenza di occhi chiusi o socchiusi e la presenza di occhiali.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{images/last_bin_1.png}
    \includegraphics[width=0.45\textwidth]{images/last_bin_2.png}
    \includegraphics[width=0.45\textwidth]{images/last_bin_3.png}
    \includegraphics[width=0.45\textwidth]{images/last_bin_4.png}
    \includegraphics[width=0.45\textwidth]{images/last_bin_5.png}
    \caption{Le cinque istanze in cui il modello finale ha dato i peggiori risultati}
\end{figure}
```

## Addestramento del classificatore per il secondo problema

L'addestramento del secondo classificatore si è svolto in maniera analoga a quello del primo. Questo significa che le *feature* utilizzate per ciascuna immagine sono sempre le stesse, cambia solamente l'etichetta utilizzata per ciascuna delle coppie di immagini, dato che si utilizza il *dataset* inerente al secondo problema.

Come prima cosa perciò si è costruito il *dataset* delle coppie di *training* per il secondo problema, dove ogni immagine è sostituita con le *feature* corrispondenti. Viene perciò creata una matrice con tante righe quante sono le istanze dalle quali è stato possibile estrarre tutte le *feature*, ma con un numero di colonne che è doppio rispetto al numero di *feature* per immagine. Nel *dataset* originale si trattengono solamente le coppie di immagini per cui le *feature* sono state estratte da entrambe, così da avere le etichette delle classi. La stessa cosa viene fatta anche per il *dataset* di test delle coppie per il secondo problema.

La rimozione di coppie dal *dataset* di *training* ha fatto in modo di modificare le frequenze delle diverse classi. Ora la relazione più presente è quella tra fratelli maschi, precedentemente terza, seguita da quella "padre - figlio", precedentemente prima, e infine quella "fratello - sorella", precedentemente settima. La relazione "madre - figlio", in origine seconda, ora è quarta. Rimangono invece pressoché invariate le altre, ovvero "madre - figlia", "padre - figlia" e quella tra sorelle. In fondo alla classifica per frequenza troviamo sempre quelle che coinvolgono i nonni, già sottorappresentate in origine.

Non c'è stato uno sbilanciamento particolarmente significativo tra le frequenze: se si considerano i rapporti tra esse, ad esempio tra la prima con 8.000 istanze e la sesta con 4.800, si nota come sono rimasti pressoché invariati, segno che l'eliminazione è avvenuta pressoché in maniera casuale e perciò simile ad un *downsampling* del *dataset*. Per questo motivo, ci si è riservati di non effettuare ulteriori manipolazioni su questo *dataset* per correggere le frequenze.

Una distribuzione simile di frequenze, anche se, in questo caso, con le varie classi ordinate in modo differente, si riscontra anche nel *test set*. Essendo quindi un *dataset* rappresentativo per il corrispettivo *training set*, anche su questo non sono state effettuate ulteriori manipolazioni.

Questo secondo problema è un problema multiclasse dove le istanze per ciascuna di esse sono sbilanciate. Per fare in modo che questo non influenzi negativamente l'addestramento, è stato assegnato un peso a ciascuna classe inversamente proporzionale alla sua presenza nel *dataset* di *training*: più è presente e meno peso avrà nell'addestramento.

\newpage

Table: Numero di istanze per le classi di parentela nel _training set_ dopo l'estrazione delle _feature_

|label|count|
|---|---|
|bb|      8019|
|fs|      6776|
|sibs|    6484|
|ms|      6145|
|md|      5139|
|fd|      4873|
|ss|      2851|
|gmgs|     669|
|gmgd|     258|
|gfgs|     240|
|gfgd|     183|

Table: Numero di istanze per le classi di parentela nel _test set_ dopo l'estrazione delle _feature_

|label|count|
|---|---|
|ms|      1161|
|bb|      1016|
|fd|       968|
|fs|       948|
|md|       817|
|sibs|     748|
|ss|       282|
|gmgd|     103|
|gfgd|      90|
|gmgs|      65|
|gfgs|      57|

Anche per questo secondo problema viene utilizzato un modello della famiglia "XGBoost", l'unica differenza risiede nel fatto che la funzione obiettivo è ora "softmax", la stessa utilizzata nelle reti neurali quando occorre risolvere un problema di classificazione multiclasse. Questa funzione fa infatti sì che per ogni input sia restituito un vettore di output i cui valori sommino a 1, in modo tale che possano essere interpretati come le probabilità di appartenenza alle diverse classi.
L'addestramento avviene sempre utilizzando *grid search* con "Bayesian Optimization" a partire dagli stessi valori utilizzati nei casi precedenti.

L'addestramento dei modelli che facevano uso della funzione somma ha avuto come risultato il fatto che i migliori tra essi hanno uno *score* attorno al 25%, sicuramente non incoraggiante. Tra questi modelli, l'addestramento ha prediletto degli *ensemble* tendenzialmente con molti alberi, ma molto bassi, dato che l'altezza massima non supera il valore di 2. Il *learning rate* è rimasto relativamente contenuto, segno di un approccio più conservativo nell'introdurre nuovi "weak classifiers". Comportamento simile, anche se più variegato, per quanto riguarda il parametro "gamma". Da ultimo invece, non si nota una chiara tendenza per quanto riguarda la percentuale di colonne e di istanze del *dataset* di *training* da considerare per ogni nuovo albero, così come il peso minimo di ogni foglia.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la somma come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.4646    |0\.1002  |0\.0101   |2        |9           |787            |0\.5431    |0\.2497     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.7697    |0\.01    |0\.0396   |1        |10          |1000           |0\.3174    |0\.2476     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.4629    |8\.9100  |0\.0162   |1        |1           |993            |0\.6701    |0\.2473     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.0934    |0\.0111  |0\.1093   |2        |1           |467            |0\.4145    |0\.2471     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.1292    |0\.1056  |0\.0239   |1        |10          |1000           |0\.2395    |0\.2470     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Leggermente più alto lo *score* per il modello che utilizza la funzione prodotto, attorno al 26%, comunque decisamente basso. In questo caso, l'addestramento ha prediletto degli *ensemble* con molti alberi, di altezza che non supera la metà del valore previsto per il suo massimo. Il *learning rate* rimane relativamente basso, come stabile rimane la percentuale di istanze campionate per ogni albero, tra il 60% e il 70%, e la percentuale di colonne considerate del *dataset* di *training* per ogni albero, ovvero pressoché la totalità delle stesse. Più ondivago il comportamento del parametro "gamma" e del peso minimo delle foglie.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano il prodotto come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|1\.0       |2\.9846  |0\.1014   |5        |1           |1000           |0\.5864    |0\.2595     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.6377    |4\.4455  |0\.0259   |7        |1           |650            |0\.6430    |0\.2592     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.1832   |3        |10          |1000           |0\.6828    |0\.2592     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.0255  |0\.0470   |7        |10          |1000           |0\.5855    |0\.2590     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0478   |8        |1           |1000           |0\.5639    |0\.2585     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Per quanto riguarda l'addestramento dei modelli che utilizzano la funzione differenza, lo score rimane inalterato rispetto al caso precedente, così come il numero di alberi nell'*ensemble*, la profondità massima degli stessi, la percentuale di istanze campionate per ogni nuovo albero e il *learning rate*. Diversamente da prima, invece, la percentuale di colonne mantenute per albero, mediamente molto poche. Il parametro "gamma" e il il peso minimo delle foglie non mostrano nessuna tendenza particolare.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la differenza come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.06      |0\.01    |0\.0352   |5        |1           |1000           |0\.7182    |0\.2605     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |9\.0000  |0\.0278   |6        |1           |1000           |0\.6823    |0\.2604     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.3823    |0\.01    |0\.0221   |6        |10          |1000           |0\.7520    |0\.2597     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |9\.0000  |0\.0172   |8        |1           |1000           |0\.5964    |0\.2594     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.0377   |5        |10          |816            |0\.6534    |0\.2589     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

I migliori modelli addestrati che utilizzano la somma al quadrato come funzione di aggregazione hanno mediamente uno *score* del 25%. In generale, l'addestramento ha prediletto come sempre *ensemble* fatti da alberi profondi e numerosi. L'approccio seguito è rimasto tendenzialmente conservativo, con *learning rate* relativamente bassi. Diversamente dal solito, ha inciso particolarmente il peso delle foglie degli alberi, che nei migliori modelli è sempre rimasto relativamente elevato e comunque superiore a 3. Tutti gli altri parametri invece non mostrano una tendenza precisa.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la somma al quadrato come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.9061    |0\.0156  |0\.1095   |2        |3           |989            |0\.1041    |0\.2485     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |9\.0000  |0\.0396   |15       |7           |1000           |0\.5434    |0\.2475     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |9\.0000  |0\.0521   |15       |10          |561            |0\.06      |0\.2475     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.3974    |0\.0108  |0\.0913   |14       |6           |962            |0\.5812    |0\.2475     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.5500    |2\.0939  |0\.0125   |15       |10          |997            |0\.0717    |0\.2475     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

I modelli che utilizzano come funzione di aggregazione il quadrato della differenza hanno uno *score* che rimane mediamente più basso tra tutti quelli visti finora, infatti non supera mai il 22%, neanche tra i migliori. L'addestramento ha preferito *ensemble* con alberi abbastanza profondi e con un numero elevato di alberi, un valore del parametro "gamma" ridotto e un *learning rate* non alto, ma sicuramente non molto basso come in altri casi. La percentuale di colonne mantenute del *dataset* di *training* per ciascun nuovo albero è rimasta di norma elevata, così come il peso minimo delle foglie. Il numero di istanze campionate non mostra una tendenza particolare.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la differenza al quadrato come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.8878    |0\.1473  |0\.0628   |8        |1           |905            |0\.1507    |0\.2221     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.8482    |0\.0319  |0\.3334   |10       |10          |307            |0\.9478    |0\.2214     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.5319    |0\.7348  |0\.3604   |1        |10          |761            |0\.8612    |0\.2212     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.01    |0\.1929   |9        |10          |518            |0\.7973    |0\.2208     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0900   |8        |6           |364            |0\.2923    |0\.2207     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Utilizzando come funzione di aggregazione la somma di quadrati, lo score medio dei migliori modelli si aggira attorno al 24,5%. Pur con un _outlier_, i migliori 5 modelli hanno tutti *ensemble* con molti alberi, minimamente profondi e con un peso minimo delle foglie pari al massimo possibile. Inoltre, la percentuale di istanze campionate per ciascun nuovo albero è sempre pari al minimo, segno di una tendenza a voler mantenere quanto più basso possibile l'*overfitting*. Anche l'approccio nell'introdurre nuovi alberi si è mantenuto conservativo, con *learning rate* nell'ordine di $10^{-2}$. Nessuna tendenza si può individuare per il parametro "gamma", mentre la percentuale delle colonne campionate rimane mediamente elevata.

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la somma di quadrati come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.6690    |8\.2664  |0\.0996   |1        |10          |1000           |0\.06      |0\.2469     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0686   |1        |10          |1000           |0\.06      |0\.2465     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.06      |0\.0150  |0\.0600   |1        |4           |734            |0\.06      |0\.2455     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.7040    |2\.4633  |0\.0532   |1        |10          |436            |0\.06      |0\.2454     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |9\.0000  |0\.0659   |1        |10          |1000           |0\.06      |0\.2450     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

L'addestramento dei modelli che utilizzavano la funzione di aggregazione differenza di quadrati hanno ottenuto dei risultati sopra la media, circa del 25,5%, anche se rimangono deludenti. Come spesso visto in passato, si torna ad avere *ensemble* molto grandi e con alberi profondi, pesi minimi delle foglie tendenzialmente elevati, *learning rate* piccoli e percentuale di campionamento delle istanze del *dataset* di *training* ridotto. Anche la percentuale di colonne utilizzate per ogni nuovo albero è relativamente elevata, mentre nulla si può dire per il parametro "gamma".

Table: Risultati dell'addestramento dei migliori cinque modelli che utilizzano la differenza di quadrati come funzione di aggregazione delle *feature*, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|1\.0       |0\.01    |0\.01     |10       |10          |729            |0\.06      |0\.2579     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.4563    |1\.0493  |0\.0106   |11       |9           |998            |0\.0826    |0\.2563     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |0\.01    |0\.0163   |15       |10          |1000           |0\.06      |0\.2558     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.6257    |0\.0254  |0\.01     |15       |3           |1000           |0\.06      |0\.2548     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|1\.0       |9\.0000  |0\.0150   |9        |10          |707            |0\.2084    |0\.2547     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Considerando quindi solamente i migliori classificatori e riaddestrandoli sull'intero *dataset* per poter valutare le loro accuratezze, si può notare, eccetto che per la funzione di somma e di somma di quadrati, un forte *overfitting*. Infatti, le accuratezze di *training* si aggirano tra il 70% e il 95%, mentre quelle di *test* si aggirano attorno al 20-25%. Anche per le funzioni citate, comunque, lo *score* di training è rispettivamente del 44% e del 54%. Anche se i modelli sono migliori di uno che tenta semplicemente una risposta casuale, bisogna considerare che le classi non sono tra di loro bilanciate e perciò possono semplicemente avere appreso le distribuzioni delle classi. Per quanto riguarda l'osservazione delle *confusion matrix*, si può notare come le parentele inerenti ai nonni, quelle con meno istanze di tutte, non vengono praticamente mai classificate come tali, né succede praticamente mai che un'istanza venga classificata come di queste classi, segno che sono state completamente snobbate. Le classi di norma più correttamente classificate sono "padre-figlio", "madre-figlia", "padre-figlia", "madre-figlio", quelle più presenti. Un'altra classe che alcuni modelli come quelli che utilizzano la differenza e la differenza di quadrati riescono ad individuare in un numero di istanze compatibile con le altre è quella dei "fratelli maschi", la più presente nel *training set*. Nonostante questi risultati incoraggianti, è pur sempre vero che le classi citate sono anche quelle per cui la rete si confonde di più e vi classifica istanze che non vi appartengono.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/sum_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 44\%, quella di 
    \textit{test} 20\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/prod_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 98\%, quella di 
    \textit{test} 23\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/diff_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 96\%, quella di 
    \textit{test} 27\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/squared_sum_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 73\%, quella di 
    \textit{test} 22\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/squared_diff_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 99\%, quella di 
    \textit{test} 20\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/sum_squares_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 54\%, quella di 
    \textit{test} 22\%}
\end{figure}
```

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/diff_squares_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello ottenuto a partire dalla funzione somma. L'accuratezza di \textit{training} era 72\%, quella di 
    \textit{test} 26\%}
\end{figure}
```

Analogamente a quanto fatto nel primo problema, sono stati costruiti i *dataset* che devono essere utilizzati dal modello XGBoost finale, che farà da "ensemble classifier" dei migliori modelli per ciascuna funzione di aggregazione dei dati individuati in precedenza. Una volta fatto, lo addestriamo nello stesso modo in cui abbiamo addestrato il modello finale del primo problema, quindi tarando gli stessi iperparametri scegliendoli tra gli stessi potenziali valori usando la medesima tecnica di "Bayesian Optimization" che in precedenza.

In questo caso, l'addestramento del modello finale ha un risultato che indica chiaramente la presenza di *overfitting*, come la sua controparte per il problema binario. Lo *score* è infatti sempre superiore al 98%. Se si vanno ad osservare gli altri parametri, si nota infatti per tutti un comportamento incerto senza tendenze precise. I parametri più "stabili" sono il peso minimo delle foglie, relativamente basso, il parametro "gamma", anch'esso relativamente basso, e il numero di alberi per modello, elevato.

Table: Risultati dell'addestramento dei migliori cinque modelli finali, l'ultima colonna rappresenta la media dello score sui tre *split* della *cross validation*

+-----------+---------+----------+---------+------------+---------------+-----------+------------+
| colsample | gamma   | learning | max     | min\_child | n\_estimators | subsample | mean\_test |
| \_bytree  |         | \_rate   | \_depth | \_weight   |               |           | \_score    |
+===========+=========+==========+=========+============+===============+===========+============+
|0\.4468    |4\.6909  |0\.0235   |13       |7           |968            |0\.4286    |0\.9831     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.8968    |0\.0499  |0\.3585   |4        |3           |221            |0\.9168    |0\.9831     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.1335    |0\.0347  |0\.4341   |9        |4           |834            |0\.2299    |0\.9831     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.0638    |8\.6022  |0\.8587   |3        |2           |111            |0\.8765    |0\.9831     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+
|0\.9005    |0\.0142  |0\.8782   |2        |1           |947            |0\.9712    |0\.9831     |
+-----------+---------+----------+---------+------------+---------------+-----------+------------+

Se si analizza l'accuratezza di quest'ultimo modello, si notano risultati analoghi a quelli dei modelli precedenti: un'accuratezza di *test* attorno al 20% e una di *training* attorno al 99%, chiaro segno di *overfitting*. Le classi più correttamente classificate rimangono le relazioni di parentela "padre-figlio", "madre-figlio", "madre-figlia" e "padre-figlia", che però sono anche le classi dove vengono più classificate istanze estranee. Le classi che riguardano i nonni vengono dal modello completamente ignorate.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{images/final_mul_cm.png}
    \caption{\textit{Confusion matrix} per il miglior modello finale. L'accuratezza di \textit{training} era 99\%, quella di \textit{test} 20\%}
\end{figure}
```

Similmente al modello costruito per il primo problema le *feature* meglio rappresentative sono la forma del viso, il colore degli occhi e della pelle. Anche in questo caso la luminosità delle immagini, la loro risoluzione e la posa del volto sembrano ricoprire un ruolo importante per una corretta classificazione.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{images/top_mul_1.png}
    \includegraphics[width=0.45\textwidth]{images/top_mul_2.png}
    \includegraphics[width=0.45\textwidth]{images/top_mul_3.png}
    \includegraphics[width=0.45\textwidth]{images/top_mul_4.png}
    \includegraphics[width=0.45\textwidth]{images/top_mul_5.png}
    \caption{Le cinque istanze in cui il modello finale ha dato i migliori risultati}
\end{figure}
```

In questo caso peculiari sono le casistiche errate, in cui è presente una grande quantità di bambini. Questo è probabilmente riconducibile alle loro caratteristiche androgine, che rendono difficoltoso distinguere i bambini dalle bambine. Da notare anche il fatto che il modello classifica con più probabilità i volti come maschili, probabilmente perché gli uomini sono più presenti nel *dataset*. Sicuramente incidono anche le differenze di luminosità, risoluzione e la presenza di immagini con occhi chiusi o socchiusi.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{images/last_mul_1.png}
    \includegraphics[width=0.45\textwidth]{images/last_mul_2.png}
    \includegraphics[width=0.45\textwidth]{images/last_mul_3.png}
    \includegraphics[width=0.45\textwidth]{images/last_mul_4.png}
    \includegraphics[width=0.45\textwidth]{images/last_mul_5.png}
    \caption{Le cinque istanze in cui il modello finale ha dato i peggiori risultati}
\end{figure}
```
