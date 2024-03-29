# Estrazione di feature handcrafted

Si è deciso di basare l'estrazione delle feature *handcrafted* sul procedimento che utilizzano gli esseri umani stessi per riconoscere un volto e, perciò, per associare due persone attraverso il loro legame di parentela. Gli elementi su cui tutti noi ci focalizziamo per analizzare un volto sono il colore della pelle, il colore e la forma degli occhi, la forma delle sopracciglia, la forma del naso, la forma della bocca e la *texture* del viso.

## Preprocessing dei dataset

Prima di effettuare l'estrazione delle *feature* di interesse, si è cercato di correggere le eventuali variazioni che possono intercorrere tra le diverse immagini. Come già detto, infatti, le immagini sono state catturate da camere diverse, a diverse angolazioni, in diverse condizioni di illuminazione, a risoluzioni diverse e con un numero diverso di canali.
Molte di queste differenze non possono essere rimosse, però si è cercato di mitigare la diversità nel numero di canali e di illuminazione. La prima, trattenendo solamente le immagini a colori ed eliminando quelle in *grayscale*, visto che si dovrà estrarre successivamente delle *feature* sui colori dell'immagine. La seconda, attraverso una correzione di gamma.

Sono state perciò caricate le immagini di *training* e di *test* dalle rispettive cartelle e sono state trattenute solamente quelle che hanno almeno un pixel il cui valore di luminosità per un canale differisce dagli altri due nella stessa posizione. Nello spazio colore "BGR", infatti, i grigi si ottengono reduplicando gli stessi valori di luminosità su tutti e tre i canali "blu", "verde" e "rosso". Se un'immagine possiede almeno un pixel con un valore di luminosità differente per uno dei canali, allora l'immagine non è più *grayscale*. Così facendo, delle 20.700 immagini del *training set* ne rimangono solamente 19.000 circa e delle 5.000 del *test set* ne rimangono solamente 4.600.

La correzione di gamma è l'applicazione di una funzione non lineare a tutti i pixel dell'immagine, che si preoccupa di equalizzare i diversi valori di luminosità. Questa dipende infatti da un parametro omonimo per cui, se $\gamma > 1$, i toni più chiari sono scuriti, mentre se $\gamma < 1$, i toni più scuri vengono schiariti.
L'idea è identificare automaticamente per ciascuna immagine il valore del parametro e la scelta è ricaduta su di un metodo utilizzato anche in alcuni software di *photoediting*. Questa tecnica consiste nel calcolare gamma come il rapporto del logaritmo della media della luminosità di tutti i pixel fratto il logaritmo di 128. In questo modo, nell'immagine la luminosità media apparirà come 50% luminosa e tutte le altre saranno tarate in conseguenza.
Poiché i pixel sono nello spazio colore BGR, viene prima fatta una conversione nello spazio HSV, corretto il canale inerente alla luminosità, ovvero "Value", e poi le immagini sono riconvertite in BGR.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.25\textwidth]{images/before_gamma.png}
    \includegraphics[width=0.25\textwidth]{images/after_gamma.png}
    \caption{Un'immagine del \textit{dataset} originale con a fianco la sua versione ``gamma corrected''}
\end{figure}
```

## Forma del naso

Per quanto riguarda il naso, la *feature* più interessante è certamente la sua forma. Anche in questo caso, la prima cosa che è stato necessario fare è quella di individuare la parte dell'immagine che la contiene.
Il naso si trova sempre tra gli occhi, limitato dagli estremi di questi a destra e a sinistra, perciò rispetto alla posizione di questi ultimi sono stati aggiunti o rimossi tanti pixel quanti ne conta il semiasse maggiore dell'ellisse che modella l'occhio. L'estremo superiore della bounding box è sempre la linea degli occhi, mentre invece l'estremo inferiore deve cadere tra la bocca e il naso, quindi all'incirca ai 15/24 dell'altezza dell'immagine.

Per estrarre la forma del naso, prima di tutto si converte l'immagine in *grayscale*, poi la si ritaglia all'altezza del naso come già detto e in seguito si applica un filtro per la riduzione del rumore, ovvero un filtro di *blur* gaussiano. Il filtro deve essere particolarmente grande, quindi forte, perché il processo di isteresi del "Canny edge detector" utilizzato subito dopo dovrà utilizzare dei valori molto bassi. Questo perché le variazioni di luminosità che indicano i contorni del naso sono molto ridotte. È stato trovato che con un filtro di dimensioni 3x9 e sigma 0 i risultati sono migliori, perché i bordi orizzontali sono facili da individuare, mentre quelli verticali si perdono facilmente. I valori per il Canny edge detector che si sono rivelati migliori sono 15 e 30 e si è utilizzato il gradiente più raffinato data la complessità di estrazione dei bordi.

Per migliorare l'immagine binaria ottenuta, si è ricorso ad una serie di operazioni di morfologia matematica per colmare le lacune tra i bordi ed ottenere perciò un contorno unico. Si è operata prima un'operazione di chiusura per colmare le lacune che si trovano tra i frammenti di bordo, anche molto grandi a causa della difficoltà riscontrata dall'operazione di "edge detection". Si è infatti adoperato un elemento strutturante 7x7 ellissoidale, in modo che i contorni avessero un aspetto più naturale. Dopodiché, si è effettuata un'operazione di dilatazione dei contorni per congiungerli tra di loro, seguita da una di erosione per riportarli al loro spessore originale. L'elemento strutturante utilizzato in questo caso, per entrambe le operazioni, è un ellisse 3x3.

Una volta ottenuti i contorni del naso finali, è stato possibile estrarre le *feature* di interesse.
Innanzitutto sono stati eliminati i bordi ancora non connessi, che a questo punto possiamo considerare come non appartenenti al naso. Questo è stato fatto attraverso l'operazione di etichettatura delle componenti connesse, andando a tenere solamente quella con area, cioè numero di pixel, maggiore.

Dopodiché si sono estratte le coordinate del centroide, nonché, sapendo le dimensioni della bounding box, l'elongazione. Si è calcolato il *convex hull* e con questo si è ottenuta la convessità della forma del naso. Queste informazioni sono utili per capire la lunghezza del naso, se è più lungo o più corto, nonché la sua tipologia, ovvero se è più "aquilino" o "schiacciato".

Infine, per non perdere completamente tutte le informazioni sulla forma del naso, si è ricorso ad un indicatore più complesso come i descrittori ellittici di Fourier. Questi descrittori costruiscono un'approssimazione dei contorni mediante una somma di ellissi ruotate o allungate in diversi modi. Queste ellissi sono estratte dalla trasformata di Fourier discreta del contorno, codificato sotto forma di segmenti lineari. Come spiegato nell'articolo originale di Kuhl e Giardina, "Elliptic Fourier features of a closed contour" [@elliptic], questi descrittori sono invarianti per rotazione, scala e traslazione e sono molto compatti, dato che un ellisse utilizza solamente 4 parametri. In questo modo, è possibile utilizzare fino a 64 curve per approssimare il contorno senza generare troppe *feature*.

L'estrazione delle feature tramite "shape matrix" è stata ritenuta troppo semplice in confronto a descrittori così avanzati come gli E.F.D. a parità di numero di valori. Indicatori come "Beam Angle Statistics" e i descrittori di Fourier semplici soffrono del problema di non poter essere estratti in numero fisso. Questo significa che o è necessario aggiungere del "padding" alla sequenza dei punti che compongono il contorno o occorre campionarlo per arrivare al numero desiderato di punti. Entrambe le operazioni, che vanno ad introdurre errore nella rappresentazione, non sono affatto desiderabili. Solamente i descrittori di Fourier, essendo coefficienti di serie di Fourier, possono essere troncati dopo essere stati calcolati ottenendo ancora un risultato valido, ma approssimato. Rimane irrisolto però il problema del padding. L'utilizzo degli E.F.D. risulta perciò più semplice a parità di informazioni estratte.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.25\textwidth]{images/nose_zone.png}
    \includegraphics[width=0.25\textwidth]{images/nose_canny.png}
    \includegraphics[width=0.25\textwidth]{images/nose_closed.png}
    \includegraphics[width=0.5\textwidth]{images/nose_data.png}
    \includegraphics[width=0.2\textwidth]{images/nose_efd.png}
    \caption{Da sinistra a destra, dall'alto in basso: la zona dell'immagine associata al naso, il risultato dell'applicazione del ``Canny edge detector'', il risultato dell'applicazione delle operazioni di morfologia matematica, la componente connessa più grande, la sua \textit{bounding box} con il centroide al suo interno, il suo
    \textit{convex hull} e infine la forma come estratta attraverso gli ``elliptic Fourier descriptors''}
\end{figure}
```

## Forma della bocca

Il processo di estrazione della forma della bocca è analogo a quello per la forma del naso, in quanto il metodo per identificare i contorni è sempre lo stesso. Naturalmente, la parte dell'immagine coinvolta come i parametri dei diversi passi che compongono il processo sono differenti, ma i concetti rimangono sempre gli stessi.

In un volto, la bocca è sempre posizionata al di sotto del naso. Per questo motivo, l'estremo superiore del rettangolo che la contiene è delimitato dalla linea che in precedenza è stata utilizzata per indicare la fine della zona del naso. Poiché la bocca è molto lunga e le sue estremità si trovano sotto gli occhi, ma non completamente, per identificare i suoi limiti sono state utilizzare le posizioni degli occhi, a cui sono stati sottratti a sinistra e aggiunti a destra i 3/2 del semiasse maggiore dell'ellisse che contiene gli stessi. Si è inoltre determinato che la posizione migliore per delimitare inferiormente la zona della bocca è ai 7/8 dell'immagine.

Con l'utilizzo del "Canny edge detector" sono stati estrapolati i bordi della bocca dall'area dell'immagine identificata in precedenza. Sull'area è statp prima applicato un filtro gaussiano 3x3 con sigma 0, poi il *detector* effettivo. Le soglie di isteresi che si sono rivelate migliori sono state 40 e 60 e anche in questo caso si è ricorso al gradiente più preciso. Si sono applicate le stesse operazioni di chiusura, dilatazione ed erosione che in precedenza. In questo caso per tutte le operazioni si è usato un elemento strutturante ellissoidale 7x7.

Da ultimo, sono state eliminate tutte le componenti connesse che non erano la più grande e poi, su questa, sono stati calcolati il centroide e l'elongazione. Calcolati i contorni è stato poi possibile calcolare il *convex hull* e quindi la convessità. Per catturare anche informazioni più complesse, sono stati calcolati i descrittori ellittici di Fourier di grado 16.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.25\textwidth]{images/mouth_zone.png}
    \includegraphics[width=0.3\textwidth]{images/mouth_canny.png}
    \includegraphics[width=0.3\textwidth]{images/mouth_closed.png}
    \includegraphics[width=0.6\textwidth]{images/mouth_data.png}
    \includegraphics[width=0.2\textwidth]{images/mouth_efd.png}
    \caption{Da sinistra a destra, dall'alto in basso: la zona dell'immagine associata alla bocca, il risultato dell'applicazione del ``Canny edge detector'', il risultato dell'applicazione delle operazioni di morfologia matematica, la componente connessa più grande, la sua \textit{bounding box} con il centroide al suo interno, il suo \textit{convex hull} e infine la forma come estratta attraverso gli ``elliptic Fourier descriptors''}
\end{figure}
```

## Forma di occhi e sopracciglia

Anche per questo terzo ed ultimo caso dove si trattava di estrapolare di nuovo la forma di oggetti, in particolare degli occhi e delle sopracciglia, sono state utilizzate le stesse tecniche che in precedenza.
In questo caso sono state individuate due aree per ciascuna immagine, ciascuna corrispondente ad un singolo occhio. A partire dalla posizione degli occhi, sono stati definiti degli *offset* che permettessero di disegnare due rettangoli che permettessero di contenerli interamente. Gli estremi inferiore e superiore sono rispettivamente 40 pixel prima e 20 pixel dopo la posizione degli occhi. Per l'occhio sinistro l'estremità sinistra si trova 25 pixel prima del suo centro mentre l'estremità destra 30 pixel dopo. Per quanto riguarda l'occhio destro, l'estremità sinistra si trova 25 pixel prima, mentre quella destra 25 pixel dopo.

In seguito è stato applicato un filtro gaussiano su entrambe le sottoimmagini estratte di dimensioni 3x3 e sigma 0. Subito dopo, sono stati estratti i contorni di occhi e sopracciglia mediante "Canny edge detector" usando come soglie di isteresi 40 e 50 e un gradiente più accurato. Sono poi stati chiusi eventuali contorni scollegati tra di loro con un'operazione di chiusura morfologica e poi sono state applicate due operazioni di dilatazione ed erosione per colmare ulteriori separazioni nell'immagine. Tutte le operazioni sono state fatte usando lo stesso elemento strutturante 5x5 ellissoidale su entrambe le sottoimmagini.

Da ultimo, vengono estratti i descrittori di forma per gli occhi e le sopracciglia. In caso le operazioni precedenti siano state effettuate in modo tale da ottenere risultati sensati, la componente connessa di area più grande, quindi che contiene il maggior numero di pixel, è necessariamente l'occhio. Individuata la componente connessa, è facile trovare il centroide, l'elongazione e i suoi contorni. Dai contorni si ottiene il *convex hull* e così la convessità. Infine, vengono calcolati i descrittori ellittici di Fourier per il contorno dell'occhio come precedentemente estratto.

L'individuazione del sopracciglio è leggermente più complessa. Innanzitutto, si eliminano tutte le componenti connesse il cui centroide ha una ordinata pari o superiore all'ordinata più in alto della *bounding box* della componente connessa dell'occhio. Questo perché ci aspettiamo che il sopracciglio si trovi sempre e comunque al di sopra dell'occhio e mai sotto o all'interno di questo. Limitarsi a considerare il centroide della componente connessa e non le dimensioni della stessa permette sovrapposizioni tra le *bounding box* di queste, che è sempre possibile. Tra tutte le componenti connesse si sceglie sempre quella di area maggiore, che dovrebbe essere il risultato dell'aggregazione delle diverse parti del contorno del sopracciglio.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.25\textwidth]{images/eyes_zone.png}
    \includegraphics[width=0.3\textwidth]{images/eyes_canny.png}
    \includegraphics[width=0.3\textwidth]{images/eyes_closed.png}
    \includegraphics[width=0.5\textwidth]{images/right_eye_data.png}
    \includegraphics[width=0.2\textwidth]{images/right_eye_efd.png}
    \includegraphics[width=0.2\textwidth]{images/right_eyebrow_efd.png}
    \includegraphics[width=0.5\textwidth]{images/left_eye_data.png}
    \includegraphics[width=0.2\textwidth]{images/left_eye_efd.png}
    \includegraphics[width=0.2\textwidth]{images/left_eyebrow_efd.png}
    \caption{Da sinistra a destra, dall'alto in basso: le zone dell'immagine associate ad occhi e sopracciglia, il risultato dell'applicazione del ``Canny edge detector'', il risultato dell'applicazione delle operazioni di morfologia matematica. In seguito, la componente connessa più grande per l'occhio e per il sopracciglio, la loro \textit{bounding box} con il rispettivo centroide al suo interno, il loro \textit{convex hull} e infine la loro forma come estratta attraverso gli ``elliptic Fourier descriptors'', prima per l'occhio destro e poi per l'occhio sinistro}
\end{figure}
```

## Tono della pelle

Per identificare il tono del colore della pelle, vengono prima segmentati i pixel che appartengono alla pelle del viso. Per farlo, viene utilizzata una tecnica basata su regole applicate agli spazi colore HSV e YCbCr. Il metodo utilizzato è quello documentato nell'articolo "Zero-sum game theory model for segmenting skin regions" [@skin] di Dahmani, Cheref e Larabi. È uno tra i metodi che viene confrontato con quello proposto dagli autori nell'articolo e su alcuni dataset fornisce risultati anche migliori di quest'ultimo. Essendo un metodo molto rapido e dalle prestazioni ottime, è stato perciò scelto come candidato.

Il metodo si basa semplicemente sul convertire l'immagine sia nello spazio colore HSV che in quello YCbCr e poi sul trattenere tutti e soli i pixel con valore compreso in specifici intervalli. Dopodiché, le due immagini vengono messe in "and" bit a bit per ottenere il risultato finale. I valori per i diversi intervalli sono:

* 0 <= H <= 17
* 15 <= S <= 170
* 135 <= Cb <= 180
* 85 <= Cr <= 135

Poiché il metodo non dà risultati perfetti, nell'implementazione fornita da Cheref vengono effettuate delle operazioni di morfologia matematica sulle maschere ottenute, in particolare di apertura con elemento strutturante rettangolare 3x3. Sulla maschera finale, viene applicato prima un filtro di "median blur" 3x3 e poi un'ulteriore operazione di apertura con elemento strutturante rettangolare 4x4.

Una volta segmentati i pixel del volto, è stato determinato il colore della pelle. L'idea è stata quella per la quale i diversi colori possano essere messi su di una scala dove il parametro che varia è la luminosità degli stessi, mentre il contenuto cromatico rimane essenzialmente invariato. Di fatto, il colore della pelle è dato dalla maggiore o minore presenza di un pigmento all'interno della stessa, che determina il suo essere più scura o più chiara, senza modificare le altre proprietà del colore, perciò variando luminosità mantenendo tinta e saturazione invariate. Questa ipotesi è supportata dal fatto che il metodo precedente di segmentazione del volto impone vincoli sulle componenti di "hue" e "saturation" nello spazio HSV e sulle componenti "Cb" e "Cr" nello spazio "YCbCr", ma lascia libero di variare "value" e "Y", che rappresentano non a caso la luminosità dell'immagine.

Per implementare questa idea, sono stati utilizzati dei campioni di colore pensati appositamente per rappresentare i diversi toni del viso. Questi campioni provengono dalla palette di colori "Monk skin tone scale" [@monk] sviluppata da Google per migliorare la rappresentazione delle diverse etnie nei suoi prodotti, pertanto ci si aspetta che sia capace di coprire con sufficiente accuratezza l'intero spettro dei colori facciali. I colori sono stati direttamente rappresentati nello spazio colore CIE La\*b\* per facilitare le operazioni successive.

Fissata la palette di riferimento, non è stato fatto altro che utilizzare i pixel del volto estratti al passo precedente per convertirli in virgola mobile, così da poter applicare la trasformazione nello spazio colore La\*b\*. Per ogni tonalità nella palette, si è costruito un *array* con stessa dimensione del numero di pixel estratti di ciascun volto, in modo tale da poter applicare la formula cosiddetta $\Delta E^\ast$ per la differenza tra colori.

Questa formula si applica infatti nello spazio La\*b\* perché è pensato per meglio rappresentare la vicinanza dei colori da un punto di vista percettivo umano e perciò alcune metriche come appunto la distanza hanno maggiore significato. Inoltre, la formula è stata pensata e studiata per evidenziare le differenze anche impercettibili tra diverse tinte di uno stesso prodotto a livello industriale, perciò vuole essere particolarmente robusta. La variante impiegata è la CIE 2000, ovvero la più raffinata della famiglia di formule $\Delta E^\ast$.

Una volta determinata la distanza tra il colore di ciascun pixel e il colore della palette attualmente in esame, si prende la media tra queste. Minimi e massimi non sono affidabili, perché potrebbero essere influenzati da pixel spuri, come anche macchie della pelle o riflessi. Il valore medio aiuta a ridurre il peso di questi *outlier*, rimanendo comunque spostato verso il valore della maggioranza dei pixel. Il descrittore è poi costruito dalla concatenazione delle diverse medie.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.25\textwidth]{images/after_gamma.png}
    \includegraphics[width=0.25\textwidth]{images/skin_tone.png}
    \caption{Un'immagine del \textit{dataset} originale con a fianco la tonalità della pelle associata}
\end{figure}
```

## Colore degli occhi

Per quanto riguarda il colore degli occhi, di norma ha una tonalità ben diversa da quelli circostanti, rendendolo un descrittore relativamente facile da estrarre. In questo caso però non possiamo assumere che il loro colore sia dato dalla maggiore o minore presenza di un pigmento al loro interno perché, oltre a non essere vero, quelle che sono considerate le principali varianti, ovvero "marroni", "verdi" e "azzurri", hanno tonalità molto diverse tra di loro. Inoltre, il colore di un occhio ha molte sfumature date dalla sua conformazione interna, anche se in questo caso non possono certo essere apprezzabili.

Per prima cosa, è stato necessario andare ad individuare la posizione degli occhi, che in tutte le immagini si trova sempre all'incirca nella stessa posizione. Questi si trovano all'incirca ad un'altezza che è i 7/24 dell'immagine. L'occhio sinistro si trova ai 7/24 della larghezza dell'immagine, mentre quello destro ai 2/3.

Questo ellisse è stato poi utilizzato per determinare una maschera da applicare sull'immagine in *grayscale* per estrapolare i pixel degli occhi. Sull'immagine con applicata la maschera sono stati poi posti tutti i pixel esterni ad essa ad una luminosità pari a 255 e poi è stato applicato *thresholding* globale binario inverso, che perciò pone a 255 tutti i pixel al di sotto di una certa soglia. La soglia, scelta manualmente, è pari a 120. Questo ha prodotto una maschera pensata per estrarre solamente i pixel dell'iride ed eliminare tutti quelli associati alla sclera, ovvero la parte bianca dell'occhio, più chiara.

Riutilizzare la tecnica precedente, ovvero quella di segmentazione dei pixel del viso non avrebbe dato risultati soddisfacenti. Purtroppo i pixel che compongono l'occhio sono molto pochi ed eliminarne qualcuno di troppo avrebbe potuto compromettere effettivamente il funzionamento del descrittore. Dato che la tecnica di segmentazione trattiene molto spesso pixel che non fanno parte del viso vero e proprio, quindi anche degli occhi, creare una maschera "al contrario" avrebbe significato quasi sicuramente perdere pixel che invece dell'occhio fanno parte. Visto che comunque utilizzando la tecnica descritta si restringe molto il campo dei potenziali pixel dell'occhio, ci si è limitati al tentativo di eliminare la sclera.

La maschera è stata poi applicata all'immagine per estrarre il *color histogram* dell'iride, che è stato calcolato come l'istogramma sul canale "hue" dell'immagine nello spazio colore "HSV".
Un istogramma calcolato solamente sul canale "hue" è sufficiente perché, in primo luogo, questo ci rende indipendenti dalla luminosità dell'immagine, che certamente influenza il colore dell'occhio. In secondo luogo, a noi interessa prevalentemente sapere la tonalità, quindi se l'occhio è più marrone, più verde o più azzurro. Per ottenere queste informazioni basta osservare come sono collocati i valori nell'istogramma: se sono presenti solamente tra il rosa e l'arancione, allora l'occhio è prevalentemente marrone, se sono presenti valori tra i verdi allora l'occhio è più verde, mentre se sono presenti valori tra i blu, allora l'occhio è azzurro. Poiché a causa del colore della pelle circostante i toni del rosso e dell'arancione rimangono sempre molto alti a prescindere dal colore dell'occhio, avrebbero potuto essere eliminati dall'istogramma. Si è però preferito lasciarli e fare in modo che sia il successivo classificatore a determinare la loro utilità o meno.
L'istogramma colore così calcolato è stato poi normalizzato: ci interessa il rapporto relativo tra le diverse tonalità e non il valore assoluto delle stesse, che dipende dal numero di pixel sui quali è stato calcolato l'istogramma e il cui valore può portare fuori strada un potenziale classificatore a causa del loro ordine di grandezza.

```{=latex}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.25\textwidth]{images/eye_zone.png}
    \includegraphics[width=0.25\textwidth]{images/eye_mask.png}
    \includegraphics[width=0.25\textwidth]{images/eye_projected.png}
    \includegraphics[width=0.5\textwidth]{images/eye_color_histogram.png}
    \caption{Da sinistra a destra, dall'alto in basso: la zona dell'immagine associata al solo interno dell'occhio, il risultato della costruzione della maschera binaria, l'applicazione della maschera sull'immagine originale e l'istogramma colore sul canale ``Hue'' come calcolato}
\end{figure}
```

## Texture del volto

Come ultimo gruppo di feature, sono state estratte quelle inerenti alla texture del volto. Queste informazioni potrebbero essere utili per catturare informazioni ancora non individuate come lentiggini, nei, rughe o altre imperfezioni del viso che possono essere trasmesse geneticamente tra parenti.

I due metodi che permettono un'estrazione "handcrafted" delle feature che risultano allo stesso tempo più veloci e capaci di codificare meglio le informazioni sono i filtri di Gabor e i "Local Binary Pattern". Tra queste due tecniche si è preferito utilizzare la seconda, nella sua variante multiscala come illustrata nel *paper* "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns" [@lbp] di Ojala, Pietikainen e Maenpaa. La sua potenza risiede nel fatto che, se si considerano solamente i *pattern* cosiddetti "uniformi e invarianti per rotazione", i quali di norma compongono fino al 90% dei pattern in un'immagine, ne esistono solamente un numero fortemente limitato. Se si indica con $P$ il numero di pixel nell'intorno considerato, allora i pattern uniformi invarianti per rotazione sono solamente $P+2$, anziché i $2^p$ totali. Questo torna utile nella tecnica che li utilizza, che consiste nel suddividere l'immagine in sottoimmagini e per ciascuna estrarre l'istogramma dei L.P.B., che conterà solamente $P+3$ *bin*, uno in più per quelli non uniformi. Quando il descrittore sarà costruito, concatenando i diversi istogrammi, la rappresentazione sarà estremamente compatta ma contenente ancora molte informazioni sulla *texture* del volto.

Per mediare tra il numero delle feature estratte e le informazioni trattenute, si è deciso di suddividere l'immagine in 16 immagini 56x56, sufficientemente piccole. Su ognuna di esse vengono calcolati gli istogrammi di L.B.P. con raggio 1, e intorno dei pixel 8, e con raggio 2, e intorno dei pixel 16, come fossero due cornici rispettivamente 3x3 e 5x5. Ciascun istogramma è stato poi normalizzato.

I filtri di Gabor non sono stati considerati adeguati perché, per ottenere dei risultati simili, avrebbero dovuto quantomeno essere generati in un banco da 2 risoluzioni per 8 direzioni, quelle tipiche della distanza Chebyshev, per un totale di 16 filtri, senza contare gli altri parametri che si possono introdurre per modificare i filtri. Dopodiché, ciascuno di essi avrebbe dovuto essere applicato su di una "saliency image" che, anche nel caso possedesse un numero basso come un centinaio di posizioni, avrebbe voluto dire 1600 *feature*, contro le 448 che estraggono così i Local Binary Pattern.
