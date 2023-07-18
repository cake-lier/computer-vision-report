# Test su immagini dei "Soliti Ignoti"

Per testare i migliori modelli addestrati in precedenza, sia tra quelli tradizionali che tra le reti neurali, sono state utilizzate le immagini prese da cinque puntate del programma televisivo dei "Soliti Ignoti". Le puntate sono tra le ultime trasmesse, ovvero quelle del 9, 11, 13, 14 e 15 aprile 2023. Per ogni puntata sono disponibili i volti degli 8 ignoti e del "parente misterioso".

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.125\textwidth]{images/relative_9.png}
    \includegraphics[width=0.825\textwidth]{images/strangers_9.png}
    \includegraphics[width=0.125\textwidth]{images/relative_11.png}
    \includegraphics[width=0.825\textwidth]{images/strangers_11.png}
    \includegraphics[width=0.125\textwidth]{images/relative_13.png}
    \includegraphics[width=0.825\textwidth]{images/strangers_13.png}
    \includegraphics[width=0.125\textwidth]{images/relative_14.png}
    \includegraphics[width=0.825\textwidth]{images/strangers_14.png}
    \includegraphics[width=0.125\textwidth]{images/relative_15.png}
    \includegraphics[width=0.825\textwidth]{images/strangers_15.png}
    \caption{Il parente misterioso e gli 8 ignoti delle cinque partite di test}
\end{figure}
```

Sono state rese disponibili anche le soluzioni in un apposito `DataFrame`. Così, per ogni episodio, corrispondente ad una specifica data, è noto quale tra gli "ignoti" era parente del "parente misterioso" e che tipo di relazione avevano i due. L'indice dell'ignoto varia tra 1 e 8, come nella trasmissione, e non tra 0 e 7, come restituisce il sistema di classificazione.

Dopodiché, sono state definite due funzioni: la prima prende in ingresso i modelli addestrati tramite tecniche di Machine Learning tradizionale sia per il primo che per il secondo problema. La seconda prende in ingresso invece quelli addestrati mediante tecniche di Representation Learning. Entrambe prendono inoltre in ingresso i concorrenti e il "parente misterioso" per effettuare un tentativo nell'individuare il concorrente corretto.
Ciascuna funzione mostra il "parente misterioso", gli 8 concorrenti e infine la soluzione proposta dalla rete e quella reale. Il modello può non avere indovinato né il parente né la parentela, solamente il parente oppure il parente e la parentela.

Quando vengono usati i modelli addestrati tramite tecniche di Machine Learning tradizionale per tentare di vincere il gioco, il sistema costruito riesce a vincere una partita, ma sbaglia completamente la classe di parentela, scambiando la classe "madre-figlio" con quella "fratelli maschi". Anche nelle partite in cui sbaglia il concorrente, il legame di parentela scelto non è mai realistico: sceglie sempre o la classe "padre-figlia" o quella "fratelli maschi", in quanto tra le due più presenti nel *dataset* di input senza un particolare criterio.

Quando vengono usati i modelli addestrati tramite tecniche di Representation Learning per tentare di vincere il gioco, il sistema riesce a vincere una partita, ma sbaglia la relazione di parentela. Scambia infatti la classe "sorelle" con quella "madre-figlio", probabilmente perché non individua correttamente il genere del concorrente. In tutti gli altri casi, però, le risposte della rete sono realistiche: nella puntata del 9 aprile predice la classe "fratello-sorella", in quella del 14 aprile predice la classe "fratelli maschi" e in quella del 15 predice la classe "sorelle", che sono legittime considerando il concorrente scelto. L'unica altra volta in cui sbaglia vistosamente la classe è nella puntata dell'11 aprile, dove predice la classe "padre-figlia", ma il parente misterioso è una donna e avrebbe dovuto essere una madre.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.45\textwidth]{images/prediction_ml_9.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_dl_9.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_ml_11.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_dl_11.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_ml_13.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_dl_13.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_ml_14.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_dl_14.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_ml_15.png}
    \includegraphics[width=0.45\textwidth]{images/prediction_dl_15.png}
    \caption{Sulla colonna di sinistra, le predizioni fatte dalla funzione che impiega i modelli ottenuti tramite tecniche di Machine Learning tradizionale, sulla colonna di destra, le predizioni fatte dalla funzione che impiega le due reti neurali}
\end{figure}
```
