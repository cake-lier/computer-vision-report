# Descrizione e analisi del dataset

Nel dataset, le immagini sono prese da diverse angolazioni, in diverse condizioni di illuminazione, da camere diverse, a diverse risoluzioni e con un diverso numero di canali. Nonostante questo, sono tutte immagini già ritagliate per contenere unicamente la faccia della persona, il che significa che non è necessario doverla estrarre.
Tutte le immagini hanno la stessa dimensione, ovvero 224 pixel per 224, e sono state caricate in memoria per essere a colori, perciò usano tre canali.

Le informazioni utili estratte dalle etichette sono state raccolte in quattro *dataset*.
I primi due sono utili a risolvere il primo problema, quello sull'individuazione dell'esistenza della parentela o meno. Questi *dataset* devono quindi associare alle coppie di immagini una _label_ binaria, che è pari a 1 se c'è parentela tra le due immagini e 0 se non c'è. Dato che le informazioni che provengono dal *dataset* originale sono solamente su parentele realmente esistenti, non erano presenti casi negativi. Per poterli avere, si sono prese casualmente immagini dello stesso *dataset* che non erano accoppiate e sono state inserite in coda al *dataset* stesso, con label negativa. Tutte le coppie di immagini che invece erano già presenti, indipendentemente dalla relazione di parentela, sono state associate ad una label positiva. La generazione dei negativi è stata fatta in modo tale che le due classi abbiano lo stesso numero di istanze, per evitare sbilanciamenti.

Il primo *dataset* è quello estratto dalle etichette delle immagini di *training*, perciò i nomi delle immagini sono dati dal codice della famiglia, "underscore", codice del componente, "underscore" e infine il nome dell'immagine originale. Il *dataset* "test_binary" è invece quello estratto dalle etichette delle immagini di test, perciò i nomi delle immagini sono quelli originali. La scelta per il *dataset* di *training* di non avere più colonne, ma una sola che dà un nome univoco all'immagine, è per semplificare il salvataggio e l'ottenimento delle immagini stesse.

\newpage

Table: Prime dieci righe del _dataset_ di _training_ per il primo problema

|index|p1|p2|label|
|---|---|---|---|
|0|F0001\_MID3\_P00005\_face2\.jpg|F0001\_MID4\_P00005\_face1\.jpg|1|
|1|F0001\_MID3\_P00005\_face2\.jpg|F0001\_MID4\_P00006\_face1\.jpg|1|
|2|F0001\_MID3\_P00005\_face2\.jpg|F0001\_MID4\_P00007\_face1\.jpg|1|
|3|F0001\_MID3\_P00006\_face2\.jpg|F0001\_MID4\_P00005\_face1\.jpg|1|
|4|F0001\_MID3\_P00006\_face2\.jpg|F0001\_MID4\_P00006\_face1\.jpg|1|
|5|F0001\_MID3\_P00006\_face2\.jpg|F0001\_MID4\_P00007\_face1\.jpg|1|
|6|F0001\_MID3\_P00007\_face3\.jpg|F0001\_MID4\_P00005\_face1\.jpg|1|
|7|F0001\_MID3\_P00007\_face3\.jpg|F0001\_MID4\_P00006\_face1\.jpg|1|
|8|F0001\_MID3\_P00007\_face3\.jpg|F0001\_MID4\_P00007\_face1\.jpg|1|
|9|F0007\_MID1\_P00073\_face5\.jpg|F0007\_MID8\_P00079\_face4\.jpg|1|
|10|F0007\_MID1\_P00073\_face5\.jpg|F0007\_MID8\_P11276\_face1\.jpg|1|

Table: Prime dieci righe del _dataset_ di _test_ per il primo problema

|index|p1|p2|label|
|---|---|---|---|
|0|face3986\.jpg|face3993\.jpg|1|
|1|face3986\.jpg|face3988\.jpg|1|
|2|face3984\.jpg|face3988\.jpg|1|
|3|face3988\.jpg|face3993\.jpg|1|
|4|face3984\.jpg|face3993\.jpg|1|
|5|face3986\.jpg|face3984\.jpg|1|
|6|face311\.jpg|face300\.jpg|1|
|7|face311\.jpg|face301\.jpg|1|
|8|face311\.jpg|face299\.jpg|1|
|9|face311\.jpg|face302\.jpg|1|
|10|face2311\.jpg|face2310\.jpg|1|

Si è deciso di utilizzare "label encoding" per codificare le undici classi di parentela che sono parte di questo progetto. Queste sono esemplificate qui di seguito:

* 0: ``bb'' (Fratelli maschi)
* 1: ``fd'' (Padre - figlia)
* 2: ``fs'' (Padre - figlio)
* 3: ``gmgs'' (Nonna - nipote maschio)
* 4: ``gmgd'' (Nonna - nipote femmina)
* 5: ``gfgs'' (Nonno - nipote maschio)
* 6: ``gfgd'' (Nonno - nipote femmina)
* 7: ``md'' (Madre - figlia)
* 8: ``ms'' (Madre - figlio)
* 9: ``sibs'' (Fratello - sorella)
* 10: ``ss'' (Sorelle)

Gli ultimi due *dataset* sono quelli inerenti al secondo problema, ovvero quello di stabilire per una coppia di immagini che rappresentano due persone sicuramente imparentate tra di loro, che tipo di relazione hanno. La struttura dei due *dataset* è analoga a quella dei precedenti, cambia solamente il contenuto in termini di istanze e la colonna "label". In questo caso infatti sono presenti solamente i casi positivi dei precedenti due *dataset*, perciò la loro dimensione è la metà dei precedenti in termini di istanze. L'etichetta però indica in questo caso il tipo di relazione tra le classi e perciò ha un valore che va da 0 a 10, in accordo all'elenco mostrato in precedenza.

Table: Prime dieci righe del _dataset_ di _training_ per il secondo problema

|index|p1|p2|relation|
|---|---|---|---|
|0|F0001\_MID3\_P00005\_face2\.jpg|F0001\_MID4\_P00005\_face1\.jpg|0|
|1|F0001\_MID3\_P00005\_face2\.jpg|F0001\_MID4\_P00006\_face1\.jpg|0|
|2|F0001\_MID3\_P00005\_face2\.jpg|F0001\_MID4\_P00007\_face1\.jpg|0|
|3|F0001\_MID3\_P00006\_face2\.jpg|F0001\_MID4\_P00005\_face1\.jpg|0|
|4|F0001\_MID3\_P00006\_face2\.jpg|F0001\_MID4\_P00006\_face1\.jpg|0|
|5|F0001\_MID3\_P00006\_face2\.jpg|F0001\_MID4\_P00007\_face1\.jpg|0|
|6|F0001\_MID3\_P00007\_face3\.jpg|F0001\_MID4\_P00005\_face1\.jpg|0|
|7|F0001\_MID3\_P00007\_face3\.jpg|F0001\_MID4\_P00006\_face1\.jpg|0|
|8|F0001\_MID3\_P00007\_face3\.jpg|F0001\_MID4\_P00007\_face1\.jpg|0|
|9|F0007\_MID1\_P00073\_face5\.jpg|F0007\_MID8\_P00079\_face4\.jpg|0|
|10|F0007\_MID1\_P00073\_face5\.jpg|F0007\_MID8\_P11276\_face1\.jpg|0|

Table: Prime dieci righe del _dataset_ di _test_ per il secondo problema

|index|p1|p2|relation|
|---|---|---|---|
|0|face3986\.jpg|face3993\.jpg|0|
|1|face3986\.jpg|face3988\.jpg|0|
|2|face3984\.jpg|face3988\.jpg|0|
|3|face3988\.jpg|face3993\.jpg|0|
|4|face3984\.jpg|face3993\.jpg|0|
|5|face3986\.jpg|face3984\.jpg|0|
|6|face311\.jpg|face300\.jpg|0|
|7|face311\.jpg|face301\.jpg|0|
|8|face311\.jpg|face299\.jpg|0|
|9|face311\.jpg|face302\.jpg|0|
|10|face2311\.jpg|face2310\.jpg|0|

Non tutte le classi hanno lo stesso peso. La classe più presente di tutti è la coppia "padre - figlio" con 56.000 istanze, seguita da quella "madre - figlio" con 6.000 istanze in meno, che a sua volta è seguita da coppie di fratelli maschi con ulteriori 6.000 istanze in meno. Solo 4.000 istanze separano questa da quella "padre - figlia", così come da questa a quella "madre - figlia". Ultima classe fortemente rappresentata è quella di coppie di fratelli di genere distinto, con solo 3.000 istanze in meno rispetto alla precedente. Ci aspettiamo perciò che per quanto riguarda le classi elencate, un eventuale classificatore faccia meno fatica a identificarle correttamente, non solo per la distanza di età più ridotta. Ci sono comunque circa 23.000 istanze di differenza tra la prima e l'ultima classe citate. Molto meno rappresentate sono le classi di sorelle, solo 17.000 istanze, e quelle che riguardano i nonni e i nipoti, che oscillano tra le 3.000 e le 1.500 istanze l'una.

Table: Numero di istanze per ciascuna classe nel _dataset_ di _training_

|relation|count|
|---|---|
|fs|      54848|
|ms|      48779|
|bb|      42679|
|fd|      38421|
|md|      34061|
|sibs|    31726|
|ss|      17600|
|gmgs|     3162|
|gfgs|     2674|
|gmgd|     1657|
|gfgd|     1564|

Raggruppando le classi per super-tipologie di relazione di parentela, si nota come quella che associa genitori con figli è quella di gran lunga più rappresentata, 176.000 istanze, quasi il doppio di quella immediatamente successiva che è quella che associa tra loro i fratelli, 92.000 istanze. La classe che relaziona i nonni con i loro nipoti ha numeri trascurabili in confronto alle altre due, poco meno di un decimo di quella immediatamente precedente. Ci aspettiamo quindi che le prime due tipologie di relazioni siano facilmente individuate dai modelli, mentre l'ultima sia molto più difficile da distinguere.

Table: Numero di istanze per ciascuna super-tipologia di parentela nel _dataset_ di _training_

|relation|count|
|---|---|
|parents - children|              176109|
|siblings|                         92005|
|grandparents - grandchildren|      9057|

C'è anche uno sbilanciamento tra i generi: gli uomini sono presenti nel dataset in circa 100.000 immagini in più rispetto alle donne. Questo significa che le relazioni tra uomini saranno più facili da individuare rispetto a quelle tra donne.

Table: Numero di istanze per genere nel _dataset_ di _training_

|relation|count|
|---|---|
|males|      324054|
|females|    230288|
