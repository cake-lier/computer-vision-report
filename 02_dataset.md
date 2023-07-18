# Descrizione e analisi del dataset

Nel dataset, le immagini sono prese da diverse angolazioni, in diverse condizioni di illuminazione, da camere diverse, a diverse risoluzioni e con un diverso numero di canali. Nonostante questo, sono tutte immagini già ritagliate per contenere unicamente la faccia della persona, il che significa che non è necessario doverla estrarre.
Tutte le immagini hanno la stessa dimensione, ovvero 224 pixel per 224, e sono state caricate in memoria per essere a colori, perciò usano tre canali.

Le informazioni utili estratte dalle etichette sono state raccolte in quattro *dataset*.
I primi due sono utili a risolvere il primo problema, quello sull'individuazione dell'esistenza della parentela o meno. Questi *dataset* devono quindi associare alle coppie di immagini una _label_ binaria, che è pari a 1 se c'è parentela tra le due immagini e 0 se non c'è. Dato che le informazioni che provengono dal *dataset* originale sono solamente su parentele realmente esistenti, non erano presenti casi negativi. Per poterli avere, si sono prese casualmente immagini dello stesso *dataset* che non erano accoppiate e sono state inserite in coda al *dataset* stesso, con label negativa. Tutte le coppie di immagini che invece erano già presenti, indipendentemente dalla relazione di parentela, sono state associate ad una label positiva. La generazione dei negativi è stata fatta in modo tale che le due classi abbiano lo stesso numero di istanze, per evitare sbilanciamenti. Il primo *dataset* è quello di *training* mentre il secondo è quello di *test*.

Gli ultimi due *dataset* sono quelli inerenti al secondo problema, ovvero quello di stabilire per una coppia di immagini che rappresentano due persone sicuramente imparentate tra di loro, che tipo di relazione hanno. La struttura dei due *dataset* è analoga a quella dei precedenti, cambia solamente il contenuto in termini di istanze e la colonna "label". In questo caso infatti sono presenti solamente i casi positivi dei precedenti due *dataset*, perciò la loro dimensione è la metà dei precedenti in termini di istanze. L'etichetta però indica in questo caso il tipo di relazione tra le classi e perciò ha un valore che va da 0 a 10, in accordo al dizionario qui di seguito mostrato. Il primo *dataset* è sempre quello di *training* e il secondo quello di *test*.

\newpage

```{=latex}
\begin{lstlisting}[language=Python, caption=Dizionario che associa ciascuna classe con la corrispondente etichetta numerica]
{'bb': 0,
 'fd': 1,
 'fs': 2,
 'gmgs': 3,
 'gmgd': 4,
 'gfgs': 5,
 'gfgd': 6,
 'md': 7,
 'ms': 8,
 'sibs': 9,
 'ss': 10}
\end{lstlisting}
```

Non tutte le classi hanno lo stesso peso. La classe più presente di tutti è la coppia "padre - figlio" con 56.000 istanze, seguita da quella "madre - figlio" con 6.000 istanze in meno, che a sua volta è seguita da coppie di fratelli maschi con ulteriori 6.000 istanze in meno. Solo 4.000 istanze separano questa da quella "padre - figlia", così come da questa a quella "madre - figlia". Ultima classe fortemente rappresentata è quella di coppie di fratelli di genere distinto, con solo 3.000 istanze in meno rispetto alla precedente. Ci aspettiamo perciò che per quanto riguarda le classi elencate, un eventuale classificatore faccia meno fatica a identificarle correttamente, non solo per la distanza di età più ridotta. Ci sono comunque circa 23.000 istanze di differenza tra la prima e l'ultima classe citate. Molto meno rappresentate sono le classi di sorelle, solo 17.000 istanze, e quelle che riguardano i nonni e i nipoti, che oscillano tra le 3.000 e le 1.500 istanze l'una.

Raggruppando le classi per super-tipologie di relazione di parentela, si nota come quella che associa genitori con figli è quella di gran lunga più rappresentata, 176.000 istanze, quasi il doppio di quella immediatamente successiva che è quella che associa tra loro i fratelli, 92.000 istanze. La classe che relaziona i nonni con i loro nipoti ha numeri trascurabili in confronto alle altre due, poco meno di un decimo di quella immediatamente precedente. Ci aspettiamo quindi che le prime due tipologie di relazioni siano facilmente individuate dai modelli, mentre l'ultima sia molto più difficile da distinguere.

C'è anche uno sbilanciamento tra i generi: gli uomini sono presenti nel dataset in circa 100.000 immagini in più rispetto alle donne. Questo significa che le relazioni tra uomini saranno più facili da individuare rispetto a quelle tra donne.
